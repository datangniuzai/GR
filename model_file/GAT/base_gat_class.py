#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/3/21 14:42
# @Author : Jason.LI
# @File : base_gat_class.py
# @Software: PyCharm

from typing import Callable

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dropout, Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.utils import register_keras_serializable

import config as cf


@register_keras_serializable(package="Custom", name="SpatioTemporalGAT")
class SpatioTemporalGAT(Layer):
    """
    A Spatio-Temporal Graph Attention Network (ST-GAT) layer.
    This layer applies Graph Attention Mechanism (GAT) across spatial and temporal dimensions.
    """

    def __init__(
            self,
            attn_heads: int,
            hid_units: int,
            activation: Callable = tf.nn.elu,
            residual: bool = False,
            dropout_rate: float = 0.2,
            use_bias: bool = True,
            static_adj_matrix: tf.Tensor = None,
            batch_size: int = None,
            name: str = None,
            **kwargs
    ):
        """
        Initializes the SpatioTemporalGAT layer.

        Args:
            attn_heads (int): Number of attention heads.
            hid_units (int): Number of hidden units in each attention head.
            activation (Callable): Activation function to apply (default: tf.nn.elu).
            residual (bool): Whether to use residual connections (default: False).
            dropout_rate (float): Dropout rate for regularization (default: 0.2).
            use_bias (bool): Whether to use bias in the layer (default: True).
            static_adj_matrix (tf.Tensor): Static adjacency matrix used for the graph structure (default: None).
            batch_size (int): The batch size (default: None).
            name (str): Layer name (default: None).
            **kwargs: Additional keyword arguments passed to the parent class.
        """
        super(SpatioTemporalGAT, self).__init__(name=name, **kwargs)

        self.attn_heads = attn_heads
        self.hid_units = hid_units
        self.activation = activation
        self.residual = residual
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.dropout = Dropout(self.dropout_rate)
        self.batch_size = batch_size

        self.a = None
        self.W = None
        self.bias = None
        self.static_adj_matrix = static_adj_matrix
        self.dynamic_adj_matrix = None

    def build(self, input_shape) -> None:
        """
        Builds the weights of the layer.

        Args:
            input_shape (tf.Tensor): The shape of the input tensor.
                Expected shape is (batch_size, time_steps, channels, feature_dim).
        """
        _, time_step_dim, channels_dim, feature_dim = input_shape

        # Linear transformation weights (for feature transformation)
        self.W = self.add_weight(
            shape=(feature_dim, self.hid_units * self.attn_heads),
            initializer="glorot_uniform",
            name="W",
        )

        # Attention mechanism weights (for computing attention scores)
        self.a = self.add_weight(shape=(2 * self.hid_units,), initializer="glorot_uniform", name="a")

        # Bias (optional)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.hid_units * self.attn_heads,), initializer="zeros", name="bias")

        # Dynamically generate adjacency matrix based on the static one
        self.dynamic_adj_matrix = self.set_dynamic_matrix(self.static_adj_matrix, self.batch_size, time_step_dim, self.attn_heads)
        self.dynamic_adj_matrix = tf.constant(self.dynamic_adj_matrix, dtype=tf.float32)

        super().build(input_shape)

    @tf.function
    def call(self, input_tensor: tf.Tensor) -> tf.Tensor:
        """
        Forward pass through the SpatioTemporalGAT layer.

        Args:
            input_tensor (tf.Tensor): The input tensor of shape (batch_size, time_steps, channels, feature_dim).

        Returns:
            tf.Tensor: The output tensor after applying the GAT layer.
        """
        b, t, c, f = tf.shape(input_tensor)[0], tf.shape(input_tensor)[1], tf.shape(input_tensor)[2], \
        tf.shape(input_tensor)[3]

        # Linear transformation
        h = tf.matmul(tf.reshape(input_tensor, [b, t * c, f]), self.W)  # [b, t*C, hid_units * attn_heads]
        h = self.dropout(h)

        h = tf.reshape(h, [b, t * c, self.attn_heads, self.hid_units])  # [b, t*C, attn_heads, hid_units]

        # Compute attention scores
        h_i = tf.tile(tf.expand_dims(h, 2), [1, 1, t * c, 1, 1])  # [b, t*C, t*C, attn_heads, hid_units]
        h_j = tf.tile(tf.expand_dims(h, 1), [1, t * c, 1, 1, 1])  # [b, t*C, t*C, attn_heads, hid_units]

        concat_h = tf.concat([h_i, h_j], axis=-1)  # [b, t*C, t*C, attn_heads, 2 * hid_units]

        e = tf.nn.leaky_relu(tf.reduce_sum(concat_h * self.a, axis=-1))  # [b, t*C, t*C, attn_heads]

        # Apply adjacency matrix mask
        e = tf.where(self.dynamic_adj_matrix > 0, e, tf.constant(-1e9, dtype=e.dtype))  # Apply adjacency matrix mask

        # Normalize attention coefficients
        alpha = tf.nn.softmax(e, axis=2)  # [b, t*C, t*C, attn_heads]
        alpha = self.dropout(alpha)

        # Feature aggregation
        h_prime = tf.reduce_sum(tf.expand_dims(alpha, -1) * h_j, axis=2)  # [b, t*C, attn_heads, hid_units]
        h_prime = tf.reshape(h_prime, [b, t, c, self.attn_heads * self.hid_units])  # [b, t, C, attn_heads * hid_units]

        # Add bias
        if self.use_bias:
            h_prime += self.bias

        # Residual connection
        if self.residual:
            h_prime += input_tensor

        return self.activation(h_prime)

    def get_config(self) -> dict:
        """
        Get the configuration of the layer.

        Returns:
            dict: Configuration dictionary.
        """
        config = super().get_config()
        config.update(
            {
                "attn_heads": self.attn_heads,
                "hid_units": self.hid_units,
                "activation": self.activation,
                "residual": self.residual,
                "dropout_rate": self.dropout_rate,
                "use_bias": self.use_bias,
            }
        )
        return config


    @staticmethod
    def set_dynamic_matrix(static_matrix, b: int ,t: int, attn_heads: int) -> tf.Tensor:
        """
        Generate the dynamic adjacency matrix for spatio-temporal graph data.

        Args:
            static_matrix: Static adjacency matrix of shape [C, C].
            b (int): Batch size.
            t (int): Time step
            attn_heads(int): number of attention heads.

        Returns:
            tf.Tensor: Dynamic adjacency matrix of shape [b, t*c, t*c, 1].
        """
        c,_ = static_matrix.shape

        # 1. Generate spatio-temporal adjacency matrix [t*c, t*c]
        static_matrix = tf.cast(static_matrix, dtype=tf.float32)
        horizontal_tiled = tf.tile(static_matrix, [1, t])  # [c, t*c]
        spatio_temporal_adj = tf.tile(horizontal_tiled, [t, 1])  # [t*c, t*c]

        # 2. Generate identity matrix [t*c, t*c]
        identity_matrix = tf.eye(t * c, dtype=tf.float32)

        # 3. Combine spatio-temporal adjacency matrix with identity matrix
        spatio_temporal_adj = spatio_temporal_adj + identity_matrix
        spatio_temporal_adj = spatio_temporal_adj[tf.newaxis, :, :, tf.newaxis]  # [1, t*c, t*c, 1]
        spatio_temporal_adj = tf.tile(spatio_temporal_adj, [b, 1, 1, attn_heads])  # [b, t*c, t*c, attn_heads]

        return spatio_temporal_adj

def stgat_model_creat(batch_size,input_matrix):

    cf.model_name = "ST-GAT"

    input_layer = Input(shape=cf.feature_shape, name='input_layer')

    gat_1 = SpatioTemporalGAT(
        attn_heads=4,
        hid_units=8,
        name="gat_layer_1",
        static_adj_matrix=input_matrix,
        dropout_rate=0.3,
        batch_size=batch_size
    )(input_layer)

    # 第二层 ST-GAT
    gat_2 = SpatioTemporalGAT(
        attn_heads=2,
        hid_units=16,
        name="gat_layer_2",
        static_adj_matrix=input_matrix,
        dropout_rate=0.3,
        batch_size=batch_size
    )(gat_1)

    output_layer = Dense(3, activation="softmax",name='output_layer')(gat_2)

    model = Model(inputs=input_layer, outputs=output_layer)

    nadam_optimizer = tf.keras.optimizers.Nadam()
    model.compile(optimizer=nadam_optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    return model

if __name__ == '__main__':
    cf.config_read()
    stgat_model_creat(batch_size=32, input_matrix=np.ones((5, 5)))
