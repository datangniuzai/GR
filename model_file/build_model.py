from typing import Optional, Tuple

import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, BatchNormalization, Activation, Dropout
from tensorflow.keras.utils import register_keras_serializable


@register_keras_serializable(package="Custom", name="TCCNN")
class TCCNN(Layer):
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

        padding: str = "valid",

        strides: Tuple[int, int] = (1, 1),

        use_batch_norm: bool = True,

        dropout_rate: float = 0.3,

        **kwargs
    ):
        super(TCCNN, self).__init__(**kwargs)

        self.filters = filters

        self.kernel_size = kernel_size

        self.activation = activation

        self.padding = padding

        self.strides = strides

        self.use_batch_norm = use_batch_norm

        self.dropout_rate = dropout_rate

        self.conv = Conv2D(filters, kernel_size, strides=strides, padding=padding)

        if use_batch_norm:
            self.batch_norm = BatchNormalization()

        self.activation_layer = Activation(activation)

        if dropout_rate > 0:

            self.dropout = Dropout(dropout_rate)

    def call(self, outer_input, first_tccnn_layer=False, last_tccnn_layer=False):

        # input shape: [batch_size, time_steps, features, channels]
        batch_size, time_steps, features, channels = outer_input.shape

        if first_tccnn_layer:

            outer_input = self.circular_padding(outer_input)

        # to [batch_size,channels + kernel_size[0] - 1,features,time_steps]
        calc_input = tf.transpose(outer_input, [0, 3, 2, 1])

        outputs = []

        for i in range(time_steps):

            data_pre_time = tf.expand_dims(calc_input[:, :, :, i], axis=-1)

            # to [batch_size,channels,filters,1]
            output_pair = tf.transpose(self.conv(data_pre_time), [0, 1, 3, 2])

            outputs.append(output_pair)

        # [batch_size,channels,filters,time_steps]
        outputs = tf.concat(outputs, axis=-1)

        if self.dropout_rate > 0:
            outputs = self.dropout(outputs)

        if self.use_batch_norm:
            outputs = self.batch_norm(outputs)

        outputs = self.activation_layer(outputs)

        if last_tccnn_layer:
            outputs = tf.reduce_mean(outputs, axis=-1)  # [batch_size,channels,filters]

        else:

            outputs = tf.transpose(outputs, [0, 3, 1, 2])  # 转换为[batch_size,time_steps,channels,filters]

        return outputs

    def circular_padding(self, inputs):
        # Get the data of the front from[0: kernel_size[0] // 2] and [-(back kernel_size[0] // 2):]
        front_padding = inputs[:, :, :, : self.kernel_size[0] // 2]
        back_padding = inputs[:, :, :, -(self.kernel_size[0] // 2) :]
        padded_data = tf.concat([front_padding, inputs, back_padding], axis=3)
        # return shape [batch_size, time_steps, features,channels+kernel_size[0]-1]
        return padded_data