# coding: utf-8
# @author: Shaw
# @datetime: 2019-05-14 14:08
# @Name: train.py.py


import numpy as np
import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from model.yolov3net import yolo_body, yolo_loss, preprocess_true_boxes


def get_classes(classes_path):
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def get_anchors(anchors_path):
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


def create_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
            weights_path='model_data/yolo_weights.h5'):
    K.clear_session()
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)
    y_true = [Input(shape=(h / {0: 32, 1: 16, 2: 8}[l], w / {0: 32, 1: 16, 2: 8}[l], num_anchors /3, num_classes + 5)
                    for l in range(3))]

    model_body = yolo_body(image_input, num_anchors ,num_classes)
    
    if load_pretrained :
        pass

def _main():
    annotation_path = 'train_txt'
    log_dir = 'logs/000/'
    classes_path = 'model_data/voc_classes.txt'
    anchors_path = 'model_data/yolo_anchors.txt'

    class_names = get_class(classes_path)
    num_classes = len(class_names)