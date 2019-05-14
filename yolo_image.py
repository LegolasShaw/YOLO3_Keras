# coding: utf-8
# @author: Shaw
# @datetime: 2019-05-08 16:50
# @Name: yolo_image.py

import os
import numpy as np
from keras.models import load_model
import colorsys
import keras.backend as k
from model.yolov3net import yolo_eval
from keras.utils import multi_gpu_model
from timeit import default_timer as timer

from PIL import Image, ImageFont, ImageDraw


def letterbox_image(image, size):
    iw, ih = image.size
    w, h = size

    scale = min(w/iw, h/ih)

    nw = int(scale * iw)
    nh = int(scale * ih)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128, 128, 128))

    new_image.paste(image, ((w-nw)/2, (h-nh)/2))
    return new_image


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
        self.sess = k.get_session()
        self.boxes, self.scores, self.classes = self.generate()

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

        self.input_image_shape = k.placeholder(shape=(2,))

        if self.gpu_num > 2:
            self.yolo_model = multi_gpu_model(self.yolo_model, self.gpu_num)
        boxes, scores, classes = yolo_eval(yolo_net_output=self.yolo_model.output, anchors=self.anchors,
                                           num_class=num_classes, image_shape=self.input_image_shape,
                                           score_threshold=self.scores, iou_threshold=self.iou)

        return boxes, scores, classes

    def detect_image(self, image):
        start = timer()

        if self.model_image_size != (None, None):
            # 断言检测
            assert self.model_image_size[0] % 32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1] % 32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width - image.width % 32),
                              image.height - (image.height - image.height % 32))

            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        image_data /= 255.0
        image_data = np.expand_dims(image_data, 0)
        out_boxes, out_scores, out_classes = self.sess.run([self.boxes, self.scores, self.classes],
                                                           feed_dict={self.yolo_model.input: image_data,
                                                                      self.input_image_shape: [image.size[1], image.size[0]],
                                                                      k.learning_phase(): 0})
        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))
        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                                  size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))

        thickness = (image.size[0] + image.size[1]) // 300

        for i, c in reversed(list(enumerate(out_classes))):  # 输出对应类型及对应索引
            predicted_class = self.class_names[c]  # 预测类型
            box = out_boxes[i]  # 检验框
            score = out_scores[i]  # 得分

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            print(label, (left, top), (right, bottom))
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for n in range(thickness):
                draw.rectangle([left + n, top + n, right + n, bottom + n], outline=self.colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
        end = timer()
        print(end - start)
        return image

    def close_seeion(self):
        self.sess.close()


if __name__ == "__main__":
    out_classes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    for i, c in reversed(list(enumerate(out_classes))):
        print(i, c)


