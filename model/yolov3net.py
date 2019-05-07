# coding: utf-8
# @author: Shaw
# @datetime: 2019-05-07 14:36
# @Name: yolov3net.py

from model.darknet53 import *

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

    pass



if __name__ == "__main__":
    inputs = Input(shape=(416, 416, 3))
    model = yolo_body(inputs=inputs, num_anchors=3, num_classes=80)
    print(K.shape(model.output[0]))
    print(K.shape(model.output[1]))
    print(K.shape(model.output[2]))