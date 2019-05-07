# coding: utf-8
# @author: Shaw
# @datetime: 2019-05-07 14:36
# @Name: yolov3net.py

from model.darknet53 import *
import os
import numpy as np

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

def yolo_eval(yolo_net_output, anchors, num_class, image_shape, score_threshold=0.6,
              iou_threahold=0.5):

    """

    :param yolo_net_output:  yolov3网络网络的输出 [[m, 13, 13, 255], [m, 26,26, 255], [m, 52,52, 255]] 255 = 3 * (80 + 5)
    :param anchors:  描点框
    :param num_class: 分类 类别数量
    :param image_shape: 输入 图像shape
    :param score_threshold:  置信度 阀值
    :param iou_threahold: 交并比阀值
    :return:
    """
    num_layers = len(yolo_net_output)
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [1, 2, 3]]
    input_shape = K.shape(yolo_net_output[0])[1:3] * 32

    boxes = [] # 预测框集合
    box_scores = [] # 预测框置信度集合

    for i in range(num_layers):
        _boxes, _box_scores = yolo_boxes_and_scores(feats=yolo_net_output[i], anchors=anchors[anchor_mask[i]],
                                                    num_classes=num_class, input_shape=input_shape,
                                                    image_shape=image_shape)
    pass


def yolo_boxes_and_scores(feats, anchors, num_classes, input_shape, image_shape):
    """
    :param feats:
    :param anchors:
    :param num_classes:
    :param input_shape:
    :param image_shape:
    :return:
    """
    box_xy, box_wh, box_confidence, box_class_prob = yolo_head(feats,
        anchors, num_classes, input_shape)
    return box, box_scores

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
    pass
    def _get_anchors():
        anchors_path = os.path.expanduser('../model_data/yolo_anchors.txt')
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    anchors = _get_anchors()[[6, 7, 8]]  # 获取最大的先验框

    anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, 3, 2])

    print(anchors_tensor)
    # inputs = Input(shape=(416, 416, 3))
    # model = yolo_body(inputs=inputs, num_anchors=3, num_classes=80)
    #
    # model.summary()
    # grid_shape = K.shape(model.output[0])[1:3]  # height, width
    # print(grid_shape)