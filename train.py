# coding: utf-8
# @author: Shaw
# @datetime: 2019-05-14 14:08
# @Name: train.py.py



from keras.callbacks import TensorBoard, ModelCheckpoint
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
    y_true = [Input(shape=(h / {0: 32, 1: 16, 2: 8}[l], w / {0: 32, 1: 16, 2: 8}[l], num_anchors /3, num_classes + 5)) for l in range(3)]

    model_body = yolo_body(image_input, num_anchors ,num_classes)

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=False, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))

        if freeze_body in [1, 2]:
            num = (20, len(model_body.layers) - 2)[freeze_body - 1]
            for i in range(num):
                model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

        model_loss = Lambda(yolo_loss, output_shape=(1, ), name='yolo_loss',
                            arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})\
                            ([*model_body.output, *y_true])

        model = Model([model_body.input, *y_true], model_loss)

        return model


def _main():
    annotation_path = 'train_txt'
    log_dir = 'logs/000/'
    classes_path = 'model_data/voc_classes.txt'
    anchors_path = 'model_data/yolo_anchors.txt'

    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    anchors = get_anchors(anchors_path)

    input_shape = (416, 416)

    is_tiny_version = len(anchors) == 6

    if is_tiny_version:

        pass
    else:
        model = create_model(input_shape, anchors, num_classes,
                             freeze_body=2, weights_path='model_data/yolo_weights.h5')

    logging = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                 monitor='val_loss', save_weights_only=True,
                                 save_best_only=True, period=3)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

    val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()

    np.random.seed(10101)
    np.random.shuffle(lines)  # 随机打乱
    np.random.seed(None)
    num_val = int(len(val_split) * val_split)
    num_train = len(lines) - num_val

    model.compile()

    if True:
        pass


if __name__ == "__main__":
    test = {"xx": lambda x, y : x}

    print(test["xx"](1, 2))



