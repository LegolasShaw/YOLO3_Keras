# coding: utf-8
# @author: Shaw
# @datetime: 2019-05-08 16:50
# @Name: yolo_image.py

import os
import numpy as np


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


if __name__ == "__main__":
    pass



