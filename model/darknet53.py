# coding: utf-8
# @author: Shaw
# @datetime: 2019-05-06 15:37
# @Name: darknet53.py

from keras.layers import Input, add,Concatenate
from keras.layers import Conv2D, BatchNormalization, LeakyReLU, Activation, UpSampling2D
from keras.regularizers import l2
from keras.models import Model
import keras.backend as K

def darknet():
    """
    :return: keras model
    """
    inputs = Input(shape=(416, 416, 3))
    x = darknet_base(input=inputs)
    # x = GlobalAveragePooling2D()(x)
    # x = Dense(1000, activation="softmax")(x)
    model = Model(inputs, x)
    return model

def make_last_lasyers(x, num_filters, out_filters):
    """
    :param x: input tensor
    :param num_filters: 卷积核数
    :param out_filters: 输出核数
    :return: 返回 采集值
    """
    x = conv2d_unit(x=x, filters=num_filters, kernels=(1, 1))
    x = conv2d_unit(x=x, filters=num_filters * 2, kernels=(3, 3))
    x = conv2d_unit(x=x, filters=num_filters, kernels=(1, 1))
    x = conv2d_unit(x=x, filters=num_filters * 2, kernels=(3, 3))
    x = conv2d_unit(x=x, filters=num_filters, kernels=(1, 1))
    y = conv2d_unit(x=x, filters=num_filters * 2, kernels=(3, 3))
    y = conv2d_unit(x=y, filters=out_filters, kernels=(1, 1))
    return x, y


def darknet_base(input):
    """
    :param input: (416, 416, 3) tensor
    :return: keras model
    """
    # 3*3 32个卷积核
    x = conv2d_unit(input, 32, (3, 3))

    # 3*3 64个卷积核 步长为2
    x = conv2d_unit(x, 64, (3, 3), strides=2)

    # 1 32 残差网络结构
    x = stack_residual_block(x, 32, 1)

    # 3*3 128 卷积核 步长为2
    x = conv2d_unit(x, 128, (3, 3), strides=2)

    # 2 64 残差网络结构
    x = stack_residual_block(x, 64, 2)

    # 3*3 256 卷积核 步长为2
    x = conv2d_unit(x, 256, (3, 3), strides=2)

    # 8 128 残差网络结构
    x = stack_residual_block(x, 128, 8)  # 52*52*256

    # 3 * 3 512 卷积核 步长为2
    x = conv2d_unit(x, 512, (3, 3), strides=2)

    # 8 256 残差网络结构
    x = stack_residual_block(x, 256, 8)

    # 3*3 1024 卷积核
    x = conv2d_unit(x, 1024, (3, 3), strides=2)

    # 4 512 残差网络
    x = stack_residual_block(x, 512, 4)

    return x


def conv2d_unit(x, filters, kernels, strides = 1):
    """
    :param x: 输入 张量
    :param filter_count: 卷积核数
    :param kernels: 卷积核
    :param stides: 扫描步长
    :return: 输出 张量
    """
    x = Conv2D(filters=filters,
               kernel_size=kernels,
               padding='same',
               strides=strides,
               activation='linear',
               kernel_regularizer=l2(5e-4))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    return x

def stack_residual_block(input, filters, n=1):
    """
    :param input: 张量输入
    :param filters: 卷积核数
    :param n: 残差网络 结构 重复次数
    :return:
    """
    x = input
    for i in range(n):
        x = residual_block(x, filters)
    return x


def residual_block(input, filters):
    x = conv2d_unit(input, filters, (1, 1))
    x = conv2d_unit(x, filters*2, (3, 3))
    x = add([input, x])
    x = Activation('linear')(x)
    return x


