# coding: utf-8
# @author: Shaw
# @datetime: 2019-05-07 14:36
# @Name: yolov3net.py

from model.darknet53 import *
import os
import numpy as np
import tensorflow as tf


def yolo_body(inputs, num_anchors, num_classes):
    """
    :param inputs:
    :param num_anchors:
    :param num_classes:
    :return:
    """
    # y1 13*13*num_anchors * (5 + num_classes)
    darknet53_model = Model(inputs=inputs, outputs=darknet_base(inputs))

    x, y1 = make_last_lasyers(darknet53_model.output, 512, num_anchors * (5 + num_classes))

    # 向上采样
    x = conv2d_unit(x, 256, (1, 1))
    x = UpSampling2D(2)(x)
    print(len(darknet53_model.layers))
    x = Concatenate()([x, darknet53_model.layers[167].output])
    # y2 26*26*num_anchors * (5 + num_classes)
    x, y2 = make_last_lasyers(x, 256, num_anchors * (5 + num_classes))

    # 向上采样
    x = conv2d_unit(x, 128, (1, 1))
    x = UpSampling2D(2)(x)
    x = Concatenate()([x, darknet53_model.layers[100].output])
    x, y3 = make_last_lasyers(x, 128, num_anchors*(num_classes+5))

    return Model(inputs, [y1, y2, y3])

def yolo_eval(yolo_net_output, anchors, num_class, image_shape,
              max_boxes=20,
              score_threshold=0.6,
              iou_threshold=0.5):
    """
    :param yolo_net_output:  yolov3网络网络的输出 [[m, 13, 13, 255], [m, 26,26, 255], [m, 52,52, 255]] 255 = 3 * (80 + 5)
    :param anchors:  描点框
    :param num_class: 分类 类别数量
    :param image_shape: 输入 图像shape (416, 416, 3)
    :param max_boxes: 同一种类 最大分类框数量
    :param score_threshold:  置信度 阀值
    :param iou_threshold: 交并比阀值
    :return:
    """
    num_layers = len(yolo_net_output)
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [1, 2, 3]]
    input_shape = K.shape(yolo_net_output[0])[1:3] * 32

    boxes = []  # 预测框集合
    box_scores = []  # 预测框置信度集合

    for i in range(num_layers):
        _boxes, _box_scores = yolo_boxes_and_scores(feats=yolo_net_output[i], anchors=anchors[anchor_mask[i]],
                                                    num_classes=num_class, input_shape=input_shape,
                                                    image_shape=image_shape)
        boxes.append(_boxes)
        box_scores.append(_box_scores)

    # boxes 的 shape (m, 4)
    # box_scores 的shape (m, 80)
    boxes = K.concatenate(boxes, axis=0)
    box_scores = K.concatenate(box_scores, axis=0)

    mask = box_scores >= score_threshold
    max_boxes_tensor = K.constant(max_boxes, dtype='int32')

    boxes_ = []
    scores_ = []
    classes_ = []

    for i in range(num_class):

        # 按分类 提取
        class_boxes = tf.boolean_mask(boxes, mask=mask[:, i])
        class_box_score = tf.boolean_mask(box_scores, mask=mask[:, i])
        nms_index = tf.image.non_max_suppression(
            class_boxes, class_box_score, max_boxes_tensor, iou_threshold=iou_threshold)

        class_boxes = K.gather(class_boxes, nms_index)
        class_box_score = K.gather(class_box_score, nms_index)
        classes = K.ones_like(class_box_score, 'int32') * i

        boxes_.append(class_boxes)
        scores_.append(class_box_score)
        classes_.append(classes)

    boxes_ = K.concatenate(boxes_, axis=0)
    scores_ = K.concatenate(scores_, axis=0)
    classes_ = K.concatenate(classes_, axis=0)

    return boxes_, scores_, classes_



def yolo_boxes_and_scores(feats, anchors, num_classes, input_shape, image_shape):
    """
    :param feats:
    :param anchors:
    :param num_classes:
    :param input_shape:
    :param image_shape:
    :return:
    """
    box_xy, box_wh, box_confidence, box_class_probs = yolo_head(feats,
        anchors, num_classes, input_shape)

    boxes = yolo_corrent_boxes(box_xy, box_wh, input_shape, image_shape)

    boxes = K.reshape(boxes, [-1, 4])
    box_scores = box_confidence * box_class_probs
    box_scores = K.reshape(box_scores, [-1, num_classes])
    return boxes, box_scores


def yolo_corrent_boxes(box_xy, box_wh, input_shape, image_shape):
    """

    :param box_xy: 检验框的xy
    :param box_wh: 检验框的wh
    :param input_shape: 输入 shape
    :param image_shape: 图像 shape
    :return:
    """
    box_yx = box_xy[..., ::-1]  # box_xy三维数组里面的 元素 逆序
    box_hw = box_wh[..., ::-1]  # box_wb三维数组里面的元素逆序

    input_shape = K.cast(input_shape, K.dtype(box_xy))
    image_shape = K.cast(image_shape, K.dtype(box_xy))

    new_shape = K.round(image_shape * K.min(input_shape/image_shape))
    offset = (input_shape-new_shape)/2./input_shape
    scale = input_shape/new_shape
    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes = K.concatenate([
        box_mins[..., 0:1],  # y_min
        box_mins[..., 1:2],  # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]  # x_max
    ])

    # Scale boxes back to original image shape.
    boxes *= K.concatenate([image_shape, image_shape])
    return boxes


def yolo_head(feats, anchors, num_classes, input_shape, calc_loss=False):
    """Convert final layer features to bounding box parameters."""
    num_anchors = len(anchors)
    # Reshape to batch, height, width, num_anchors, box_params.
    anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, num_anchors, 2])

    grid_shape = K.shape(feats)[1:3]  # height, width
    grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]), [1, grid_shape[1], 1, 1])
    grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]), [grid_shape[0], 1, 1, 1])
    grid = K.concatenate([grid_x, grid_y])
    grid = K.cast(grid, K.dtype(feats))

    feats = K.reshape(
        feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

    # Adjust preditions to each spatial grid point and anchor size.
    box_xy = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[::-1], K.dtype(feats))
    box_wh = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[::-1], K.dtype(feats))
    box_confidence = K.sigmoid(feats[..., 4:5])
    box_class_probs = K.sigmoid(feats[..., 5:])

    if calc_loss == True:
        return grid, feats, box_xy, box_wh
    return box_xy, box_wh, box_confidence, box_class_probs


def yolo_loss(args, anchors, num_classes, ignore_threshold=0.5, print_loss=False):
    """

    :param args:
    :param anchors: 宽和高 数组
    :param num_class: 类别数
    :param ignore_threshold:  忽略值阀值
    :param print_loss: 是否输出 loss
    :return:
    """
    num_layers = len(anchors) // 3  # default setting
    yolo_outputs = args[:num_layers]
    y_true = args[num_layers:]
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [1, 2, 3]]
    input_shape = K.cast(K.shape(yolo_outputs[0])[1:3] * 32, K.dtype(y_true[0]))
    grid_shapes = [K.cast(K.shape(yolo_outputs[l])[1:3], K.dtype(y_true[0])) for l in range(num_layers)]
    loss = 0
    m = K.shape(yolo_outputs[0])[0]  # batch size, tensor
    mf = K.cast(m, K.dtype(yolo_outputs[0]))

    for l in range(num_layers):
        object_mask = y_true[l][..., 4:5]
        true_class_probs = y_true[l][..., 5:]

        grid, raw_pred, pred_xy, pred_wh = yolo_head(yolo_outputs[l],
                                                     anchors[anchor_mask[l]], num_classes, input_shape,
                                                     calc_loss=True)
        pred_box = K.concatenate([pred_xy, pred_wh])

        # Darknet raw box to calculate loss.
        raw_true_xy = y_true[l][..., :2] * grid_shapes[l][::-1] - grid
        raw_true_wh = K.log(y_true[l][..., 2:4] / anchors[anchor_mask[l]] * input_shape[::-1])
        raw_true_wh = K.switch(object_mask, raw_true_wh, K.zeros_like(raw_true_wh))  # avoid log(0)=-inf
        box_loss_scale = 2 - y_true[l][..., 2:3] * y_true[l][..., 3:4]

        # Find ignore mask, iterate over each of batch.
        ignore_mask = tf.TensorArray(K.dtype(y_true[0]), size=1, dynamic_size=True)
        object_mask_bool = K.cast(object_mask, 'bool')

        def loop_body(b, ignore_mask):
            true_box = tf.boolean_mask(y_true[l][b, ..., 0:4], object_mask_bool[b, ..., 0])
            iou = box_iou(pred_box[b], true_box)
            best_iou = K.max(iou, axis=-1)
            ignore_mask = ignore_mask.write(b, K.cast(best_iou < ignore_threshold, K.dtype(true_box)))
            return b + 1, ignore_mask

        _, ignore_mask = K.control_flow_ops.while_loop(lambda b, *args: b < m, loop_body, [0, ignore_mask])
        ignore_mask = ignore_mask.stack()
        ignore_mask = K.expand_dims(ignore_mask, -1)

        # K.binary_crossentropy is helpful to avoid exp overflow.
        xy_loss = object_mask * box_loss_scale * K.binary_crossentropy(raw_true_xy, raw_pred[..., 0:2],
                                                                       from_logits=True)
        wh_loss = object_mask * box_loss_scale * 0.5 * K.square(raw_true_wh - raw_pred[..., 2:4])
        confidence_loss = object_mask * K.binary_crossentropy(object_mask, raw_pred[..., 4:5], from_logits=True) + \
                          (1 - object_mask) * K.binary_crossentropy(object_mask, raw_pred[..., 4:5],
                                                                    from_logits=True) * ignore_mask
        class_loss = object_mask * K.binary_crossentropy(true_class_probs, raw_pred[..., 5:], from_logits=True)

        xy_loss = K.sum(xy_loss) / mf
        wh_loss = K.sum(wh_loss) / mf
        confidence_loss = K.sum(confidence_loss) / mf
        class_loss = K.sum(class_loss) / mf
        loss += xy_loss + wh_loss + confidence_loss + class_loss
        if print_loss:
            loss = tf.Print(loss, [loss, xy_loss, wh_loss, confidence_loss, class_loss, K.sum(ignore_mask)],
                            message='loss: ')
    return loss


def box_iou(b1, b2):
    '''Return iou tensor
    Parameters
    ----------
    b1: tensor, shape=(i1,...,iN, 4), xywh
    b2: tensor, shape=(j, 4), xywh
    Returns
    -------
    iou: tensor, shape=(i1,...,iN, j)
    '''

    # Expand dim to apply broadcasting.
    b1 = K.expand_dims(b1, -2)
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh/2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half

    # Expand dim to apply broadcasting.
    b2 = K.expand_dims(b2, 0)
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh/2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    intersect_mins = K.maximum(b1_mins, b2_mins)
    intersect_maxes = K.minimum(b1_maxes, b2_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    iou = intersect_area / (b1_area + b2_area - intersect_area)
    return iou


def preprocess_true_boxes(true_box, input_shape, anchors, num_classes):
    """

    :param true_box:
    :param input_shape:
    :param anchors:
    :param num_classes:
    :return:
    """
    assert (true_box[..., :4] < num_classes).all(), 'class id must be less than num_classes'
    num_layers = len(anchors) / 3

    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [1, 2, 3]]
    true_boxes = np.array(true_box, dtype='float32')
    input_shape = np.array(input_shape, dtype='int32')

    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 3:4])/ 2
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]

    true_boxes[..., 0:2] = boxes_xy/input_shape[::-1]
    true_boxes[..., 2:4] = boxes_wh/input_shape[::-1]

    m = true_box.shape[0]
    grid_shapes = [input_shape / {0: 32, 1: 16, 2: 8} for l in range(num_layers)]
    y_true = [np.zeros(m, grid_shapes[l][0], grid_shapes[l][1], len(anchor_mask[l]), 5+num_classes, dtype='float32') for l in range(num_layers)]

    anchors = np.expand_dims(anchors, 0)
    anchor_maxes = anchors / 2
    anchor_mins = -anchor_maxes
    valid_mask = boxes_wh[..., 0] > 0

    for b in range(m):
        wh = boxes_wh[b, valid_mask[b]]
        if len(wh) == 0:
            continue
        wh = np.expand_dims(wh, -2)
        box_maxes = wh / 2
        box_mins = -box_maxes

        intersect_mins = np.maximum(box_mins, anchor_mins)
        intersect_maxes = np.minimum(box_maxes, anchor_maxes)

        intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]

        box_area = wh[..., 0] * wh[..., 1]
        anchor_area = anchors[..., 0] * anchors[..., 1]

        iou = intersect_area / (box_area + anchor_area - intersect_area)

        best_anchor = np.argmax(iou, axis=-1)

        for t, n in enumerate(best_anchor):
            for l in range(num_layers):
                if n in anchor_mask[l]:
                    i = np.floor(true_boxes[b, t, 0] * grid_shapes[l][1]).astype('int32')
                    j = np.floor(true_boxes[b, t, 1] * grid_shapes[l][0]).astype('int32')
                    k = anchor_mask[l].index(n)
                    c = true_boxes[b, t, 4].astype('int32')
                    y_true[l][b, j, i, k, 0:4] = true_boxes[b, t, 0:4]
                    y_true[l][b, j, i, k, 4] = 1
                    y_true[l][b, j, i, k, 5 + c] = 1

    return y_true

    # grid_shapes = [input_shape]

if __name__ == "__main__":
    # inputs = Input(shape=(416, 416, 3))
    # model = yolo_body(inputs=inputs, num_anchors=3, num_classes=80)
    # model.summary()
    # grid_shape = K.shape(model.output[0])[1:3]  # height, width
    # print(grid_shape)

    # tf_session = K.get_session()
    #
    # xx = np.array([0.5, 0.1, 1.0, 0.8])
    #
    # mask = xx >= 0.4
    # print(mask)
    #
    # test = np.array([1.0, 2.0, 3.0, 4.0])
    #
    # y = tf.boolean_mask(test, xx, axis=0)
    # print(y.eval(session=tf_session))

    input_shape = [416, 416]
    input_shape = np.array(input_shape)

    grid_shapes = [input_shape // {0: 32, 1: 16, 2: 8}[l] for l in range(3)]

    print(grid_shapes)