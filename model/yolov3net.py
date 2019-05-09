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


if __name__ == "__main__":
    # inputs = Input(shape=(416, 416, 3))
    # model = yolo_body(inputs=inputs, num_anchors=3, num_classes=80)
    # model.summary()
    # grid_shape = K.shape(model.output[0])[1:3]  # height, width
    # print(grid_shape)

    tf_session = K.get_session()

    xx = np.array([0.5, 0.1, 1.0, 0.8])

    mask = xx >= 0.4
    print(mask)

    test = np.array([1.0, 2.0, 3.0, 4.0])

    y = tf.boolean_mask(test, xx, axis=0)
    print(y.eval(session=tf_session))