#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2024/11/20 13:14
# @Author : 李 嘉 轩
# @File : models.py
# @Software: PyCharm

import tensorflow as tf
from keras.layers import Input, Flatten, Dense, concatenate
from keras.models import Model

import config as cf
from model_file.my_layers import TccnnV1, ChannelSelector

def TCCNN_model_creat():
    cf.model_name = "TCCNN"
    input_layer = Input(shape=cf.feature_shape, name='input_layer')

    flatten_layers = []
    for channel_index in range(4):
        selected_channels_layer = ChannelSelector(start_index=channel_index, step_size=4)(input_layer)
        tccnn1 = TccnnV1(filters=10, kernel_size=(5, 5), name=f'Tccnn{channel_index + 1}_1',
                     dropout_rate=0.3)(selected_channels_layer, concatenation_padding=True, last_tccnn=False)
        tccnn2 = TccnnV1(filters=15, kernel_size=(5, 10), name=f'Tccnn{channel_index + 1}_2',
                     dropout_rate=0.2)(tccnn1, concatenation_padding=False, last_tccnn=True)
        flatten_layer = Flatten(name=f'flatten_{channel_index + 1}')(tccnn2)
        flatten_layers.append(flatten_layer)

    # 直接使用 input_layer 作为 tccnn5 的输入
    tccnn5_1 = TccnnV1(filters=10, kernel_size=(5, 5), name='Tccnn5_1', dropout_rate=0.3)(input_layer,
                                                                                                      concatenation_padding=True,
                                                                                                      last_tccnn=False)
    tccnn5_2 = TccnnV1(filters=20, kernel_size=(5, 10), name='Tccnn5_2', dropout_rate=0.2)(tccnn5_1,
                                                                                                       concatenation_padding=False,
                                                                                                       last_tccnn=True)
    flatten5 = Flatten(name='flatten_5')(tccnn5_2)
    flatten_layers.append(flatten5)

    # 拼接部分
    concat = concatenate(flatten_layers, name='concatenate')

    # 全连接层
    dense1 = tf.keras.layers.Dense(units=120, activation='relu', name='dense_last')(concat)

    # 输出层
    output_layer = Dense(units=cf.gesture_num, activation='softmax', name='output_layer_last')(dense1)
    model = Model(inputs=input_layer, outputs=output_layer)

    # 预设置优化器
    nadam_optimizer = tf.keras.optimizers.Nadam()
    model.compile(optimizer=nadam_optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # 编译模型
    model.summary()

    return model