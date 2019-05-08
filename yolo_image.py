# coding: utf-8
# @author: Shaw
# @datetime: 2019-05-08 16:50
# @Name: yolo_image.py

import os
import numpy as np
from keras.models import load_model
import colorsys
import keras.backend as K
from model.yolov3net import yolo_eval
from keras.utils import multi_gpu_model
from timeit import default_timer as timer

from PIL import Image


class YoloClass(object):
    _defaults = {
        "model_path": "model_data/yolo.h5",
        "anchors_path": "model_data/yolo_anchors.txt",
        "classes_path": "model/coco_class.txt",
        "score": 0.3,
        "iou": 0.45,
        "model_image_size": (416, 416),
        "gpu_num": 1
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [name.strip() for name in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors_str = f.readline()
        anchors_list = anchors_str.split(',')
        anchors_list = [float(x) for x in anchors_list]
        return np.reshape(anchors_list, newshape=[-1, 2])

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        self.__dict__.update(kwargs)
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxesm, self.scores, self.classes = self.self.generate()

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5')

        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)

        is_tiny_version = num_anchors == 6  # ???

        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            pass

        # 构造 标记框 的颜色  (不同类别, 1.0, 1.0)
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))

        np.random.seed(10101)
        np.random.shuffle(self.colors)
        np.random.seed(None)

        self.input_image_shape = K.placeholder(shape=(2,))

        if self.gpu_num > 2:
            self.yolo_model = multi_gpu_model(self.yolo_model, self.gpu_num)
        boxes, scores, classes = yolo_eval(yolo_net_output=self.yolo_model.output, anchors=self.anchors,
                                           num_class=num_classes, image_shape=self.input_image_shape,
                                           score_threshold=self.scores, iou_threshold=self.iou)


        return  boxes, scores, classes

    def detect_image(self, image):
        start = timer()



def letterbox_image(image, size):
    iw, ih = image.size
    w, h = size

    scale = min(w/iw, h/ih)

    nw = int(scale * iw)
    nh = int(scale * ih)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128, 128))

    new_image.paste(image, ((w-nw)/2, (h-nh)/2))
    return new_image


if __name__ == "__main__":
    input_image_shape = K.placeholder(shape=(2,))
    print(input_image_shape)


