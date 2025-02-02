#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2024/02/01 19:43
# @Author : Jiaxuan LI
# @File : build_model.py
# @Software: PyCharm

from typing import Tuple

import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, BatchNormalization, Activation, Dropout, Input, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.utils import register_keras_serializable

import config as cf


@tf.keras.utils.register_keras_serializable(package="Custom", name="ChannelSelector")
class ChannelSelector(Layer):
    """
    A custom Keras layer that selects channels from an input tensor based on the given start_index and step_size.

    Attributes:
        start_index (int): The index where the channel selection starts.
        step_size (int): The step size used for channel selection (i.e., selecting every 'step_size' channels).
    """

    def __init__(self, start_index: int, step_size: int, **kwargs):
        """
        Initializes the ChannelSelector layer with the given parameters.

        Args:
            start_index (int): The starting index for channel selection.
            step_size (int): The step size for selecting channels.
            **kwargs: Additional arguments to be passed to the parent Layer class.
        """
        super(ChannelSelector, self).__init__(**kwargs)
        self.start_index = start_index
        self.step_size = step_size

    def call(self, outer_input: tf.Tensor, **kwargs) -> tf.Tensor:
        """
        Selects channels from the input tensor based on the start_index and step_size.

        Args:
            outer_input (tf.Tensor): The input tensor of shape [batch_size, time_steps, features, channels].
            **kwargs: Additional keyword arguments.

        Returns:
            tf.Tensor: The tensor with selected channels, shape [batch_size, time_steps, features, selected_channels].
        """
        num_channels = tf.shape(outer_input)[-1]

        selected_indices = tf.range(self.start_index, num_channels, self.step_size)
        print(selected_indices)
        selected_channels = tf.gather(outer_input, selected_indices, axis=-1)

        return selected_channels

    def get_config(self) -> dict:
        """
        Returns the configuration of the ChannelSelector layer.

        Returns:
            dict: The configuration dictionary including the start_index and step_size.
        """
        config = super(ChannelSelector, self).get_config()
        config.update({
            "start_index": self.start_index,
            "step_size": self.step_size
        })
        return config

@register_keras_serializable(package="Custom", name="TCCNN")
class TCCNNLayer(Layer):
    """
    Calculating new features by fusing adjacent time steps.
    --------
    Input:  (batch_size, time_steps, features, channels)
    Output: (batch_size, time_steps, filters , channels)
    """

    def __init__(
        self,
        filters: int,
        kernel_size: Tuple[int, int],
        activation: str = "relu",
        strides: Tuple[int, int] = (1, 1),
        use_batch_norm: bool = True,
        dropout_rate: float = 0.3,
        **kwargs
    ):
        super(TCCNNLayer, self).__init__(**kwargs)

        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = activation
        self.strides = strides
        self.use_batch_norm = use_batch_norm
        self.dropout_rate = dropout_rate

        self.conv = Conv2D(filters, kernel_size, strides=strides, padding="valid")
        if use_batch_norm:
            self.batch_norm = BatchNormalization()
        self.activation_layer = Activation(activation)
        if dropout_rate > 0:
            self.dropout = Dropout(dropout_rate)

    @tf.function
    def call(
        self,
        outer_input: tf.Tensor,
        first_tccnn_layer: bool = False,
        last_tccnn_layer: bool = False
    ) -> tf.Tensor:
        """
        Forward pass of the TCCNN layer.
        """
        if first_tccnn_layer:
            outer_input = self.circular_padding(outer_input)

        conv_input = tf.expand_dims(outer_input, axis=-1)
        conv_output_temp = self.conv(conv_input)

        temp_output = tf.squeeze(conv_output_temp, axis=2)

        if self.use_batch_norm:
            temp_output = self.batch_norm(temp_output)

        temp_output = self.activation_layer(temp_output)

        if self.dropout_rate > 0:
            temp_output = self.dropout(temp_output)

        if last_tccnn_layer:
            outputs = tf.reduce_mean(temp_output, axis=1)
        else:
            outputs = tf.transpose(temp_output, [0, 1, 3, 2])

        return outputs

    def circular_padding(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Apply circular padding along the channels dimension.
        """
        pad_size = self.kernel_size[0] // 2
        padded_data = tf.concat([
            inputs[:, :, :, -pad_size:],  # Last pad_size channels (for left padding)
            inputs,  # Original input
            inputs[:, :, :, :pad_size]  # First pad_size channels (for right padding)
        ], axis=3)
        return padded_data

    def get_config(self):
        """
        Returns the config dictionary for the custom layer.
        """
        config = super(TCCNNLayer, self).get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'activation': self.activation,
            'strides': self.strides,
            'use_batch_norm': self.use_batch_norm,
            'dropout_rate': self.dropout_rate
        })
        return config



def tccnn_model_creat():

    cf.model_name = "TCCNN"

    input_layer = Input(shape=cf.feature_shape, name='input_layer')

    tccnn_1 = (TCCNNLayer(filters=10, kernel_size=(5, 5), name='tccnn_1', dropout_rate=0.3)
               (input_layer,first_tccnn_layer=True,last_tccnn_layer=False))
    tccnn_2 = (TCCNNLayer(filters=20, kernel_size=(10, 5), name='tccnn_2', dropout_rate=0.2)
               (tccnn_1,first_tccnn_layer=False,last_tccnn_layer=True))
    flatten_1 = Flatten(name='flatten_1')(tccnn_2)

    dense1 = tf.keras.layers.Dense(units=120, activation='relu', name='dense_last')(flatten_1)

    output_layer = Dense(units=cf.gesture_num, activation='softmax', name='output_layer')(dense1)
    model = Model(inputs=input_layer, outputs=output_layer)

    nadam_optimizer = tf.keras.optimizers.Nadam()
    model.compile(optimizer=nadam_optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    return model