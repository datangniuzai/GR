#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2024/11/14 17:38
# @Author : 李 嘉 轩
# @File : tccnn.py
# @Software: PyCharm

import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, BatchNormalization, Activation, Dropout

@tf.keras.utils.register_keras_serializable(package="Custom", name="TCCNN_V1")
class TCCNN_V1(Layer):
    '''
    Calculating new features by fusing adjacent time steps.
    --------
    Input:  (batch_size, time_steps   , channels , features )
    Output: (batch_size, time_steps , channels , filters  )
    '''

    def __init__(self,
                 filters,
                 kernel_size,
                 activation='relu',
                 strides=(1,1),
                 use_batch_norm=True,
                 dropout_rate=0.3, **kwargs):

        super(TCCNN_V1, self).__init__(**kwargs)

        # 滤波器的数量
        self.filters = filters

        # 滤波器的尺寸
        self.kernel_size = kernel_size

        # 滤波器的步长
        self.strides = strides

        # 设定dropout率
        self.dropout_rate = dropout_rate

        # 初始化滤波器
        self.conv = Conv2D(filters, kernel_size, strides=strides, padding="valid")

        # 是否使用批量归一化
        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.batch_norm = BatchNormalization()

        # 初始化激活函数层
        self.activation = activation
        self.activation_layer = Activation(activation)

        # 初始化dropout层
        if dropout_rate > 0:
            self.dropout = Dropout(dropout_rate)

    def call(self,
             inputs,
             concatenation_padding = False,
             last_tccnn = False):

        # 输入形状: [batch_size, time_steps, channels, features]
        batch_size, time_steps, channels, features = inputs.shape

        # 判断是否进行拼接填充并调整形状以适应Conv2D
        if concatenation_padding:
            padded_inputs = self.preprocess_data(inputs)
            padded_inputs = tf.transpose(padded_inputs, [0, 1, 3,2])  # 转换为 [batch_size,channels + kernel_size[0] - 1,features,time_steps]
        else:
            padded_inputs = tf.transpose(inputs, [0, 2, 3, 1])  # 转换为 [batch_size,channels ,features,time_steps]

        # 进行卷积操作
        # 准备容器
        outputs = []

        # 先提取每个时间步的特征
        for i in range(time_steps):
            input_pair = padded_inputs[:, :, :, i]

            # 维持维度数量
            input_pair = tf.expand_dims(input_pair, axis=-1)

            # [batch_size,channels,1,filters]
            output_pair = self.conv(input_pair)

            # 调整顺序，等待拼接，转换为[batch_size,channels,filters,1]
            output_pair = tf.transpose(output_pair, [0, 1, 3, 2])

            # 数据装入容器
            outputs.append(output_pair)

        # 在最后一个维度上进行拼接，[batch_size,channels,filters,time_steps]
        outputs = tf.concat(outputs, axis=-1)

        # 进行dropout层优化，防止过拟合
        if self.dropout_rate > 0:
            outputs = self.dropout(outputs)

        # 批量归一化，便于优化
        if self.use_batch_norm:
            outputs = self.batch_norm(outputs)

        # 进入非线性激活层
        outputs = self.activation_layer(outputs)

        # 如果是最后一层TCCNN,就对时间维度进行平均池化
        if last_tccnn:
            outputs = tf.reduce_mean(outputs, axis=-1)  # [batch_size,channels,filters]
        else:
            outputs = tf.transpose(outputs, [0, 3, 1, 2])  # 转换为[batch_size,time_steps,channels,filters]

        return outputs

    def preprocess_data(self, inputs):

        # 获取前 kernel_size[0] // 2 和后 kernel_size[0] // 2 的数据
        inputs = tf.transpose(inputs, [0, 2, 1, 3])  # 转换为 [batch_size, channels, time_steps, features]
        front_padding = inputs[:, :self.kernel_size[0] // 2, :, :]
        back_padding = inputs[:, -(self.kernel_size[0]//2):, :, :]

        # 拼接数据
        padded_inputs = tf.concat([front_padding, inputs, back_padding], axis=1) # 转换为 [batch_size, channels+kernel_size[0]-1, time_steps, features]

        return padded_inputs


@tf.keras.utils.register_keras_serializable(package="Custom", name="TCCNN_V2")
class TCCNN_V2(Layer):
    '''
    Calculating new features by fusing adjacent time steps.
    --------
    Input:  (batch_size, time_steps, channels, features)
    Output: (batch_size, time_steps, channels, filters)
    '''

    def __init__(self,
                 filters,
                 kernel_size,
                 activation='relu',
                 strides=(1, 1),
                 use_batch_norm=True,
                 dropout_rate=0.3,
                 **kwargs):

        super(TCCNN_V2, self).__init__(**kwargs)

        # 滤波器的数量
        self.filters = filters

        # 滤波器的尺寸
        self.kernel_size = kernel_size

        # 滤波器的步长
        self.strides = strides

        # 设定dropout率
        self.dropout_rate = dropout_rate

        # 初始化滤波器
        self.conv = Conv2D(filters, kernel_size, strides=strides, padding="valid")

        # 是否使用批量归一化
        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.batch_norm = BatchNormalization()

        # 初始化激活函数层
        self.activation = activation
        self.activation_layer = Activation(activation)

        # 初始化dropout层
        if dropout_rate > 0:
            self.dropout = Dropout(dropout_rate)

    def call(self,
             inputs,
             concatenation_padding=False,
             last_tccnn=False):

        # 输入形状: [batch_size, time_steps, channels, features]
        batch_size, time_steps, channels, features = inputs.shape

        # 判断是否进行拼接填充并调整形状以适应Conv2D
        if concatenation_padding:
            padded_inputs = self.preprocess_data(inputs)
            padded_inputs = tf.transpose(padded_inputs, [0, 1, 3,
                                                         2])  # 转换为 [batch_size,channels + kernel_size[0] - 1,features,time_steps]
        else:
            padded_inputs = tf.transpose(inputs, [0, 2, 3, 1])  # 转换为 [batch_size,channels ,features,time_steps]

        # 进行卷积操作
        # 准备容器
        outputs = []

        # 先提取每个时间步的特征
        for i in range(time_steps):
            input_pair = padded_inputs[:, :, :, i]

            # 维持维度数量
            input_pair = tf.expand_dims(input_pair, axis=-1)

            # [batch_size,channels,1,filters]
            output_pair = self.conv(input_pair)

            # 调整顺序，等待拼接，转换为[batch_size,channels,filters,1]
            output_pair = tf.transpose(output_pair, [0, 1, 3, 2])

            # 数据装入容器
            outputs.append(output_pair)

        # 在最后一个维度上进行拼接，[batch_size,channels,filters,time_steps]
        outputs = tf.concat(outputs, axis=-1)

        # 进行dropout层优化，防止过拟合
        if self.dropout_rate > 0:
            outputs = self.dropout(outputs)

        # 批量归一化，便于优化
        if self.use_batch_norm:
            outputs = self.batch_norm(outputs)

        # 进入非线性激活层
        outputs = self.activation_layer(outputs)

        # 如果是最后一层TCCNN,就对时间维度进行平均池化
        if last_tccnn:
            outputs = tf.reduce_mean(outputs, axis=-1)  # [batch_size,channels,filters]
        else:
            outputs = tf.transpose(outputs, [0, 3, 1, 2])  # 转换为[batch_size,time_steps,channels,filters]

        return outputs

    def preprocess_data(self, inputs):

        # 获取前 kernel_size[0] // 2 和后 kernel_size[0] // 2 的数据
        inputs = tf.transpose(inputs, [0, 2, 1, 3])  # 转换为 [batch_size, channels, time_steps, features]

        batch_size, channels, time_steps, features = inputs.shape

        num_groups = channels // 4

        averaged_inputs = []

        for i in range(num_groups):
            group = inputs[:, i * 4:(i + 1) * 4, :, :]
            averaged_group = tf.reduce_mean(group, axis=1)
            averaged_inputs.append(averaged_group)
        averaged_inputs = tf.stack(averaged_inputs, axis=1)

        front_padding = inputs[:, :self.kernel_size[0] // 2, :, :]
        back_padding = inputs[:, -(self.kernel_size[0] // 2):, :, :]

        # 拼接数据
        padded_inputs = tf.concat([front_padding, averaged_inputs, back_padding],
                                  axis=1)  # 转换为 [batch_size, channels+kernel_size[0]-1, time_steps, features]
        return padded_inputs