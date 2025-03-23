#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/3/21 14:42
# @Author : Jason.LI
# @File : base_gat_class.py
# @Software: PyCharm

import tensorflow as tf
from tensorflow.keras.layers import Layer, Dropout


class SpatioTemporalGAT(Layer):
    def __init__(self, attn_heads, hid_units, activation=tf.nn.elu, residual=False, dropout_rate=0.2, use_bias=True):
        """
        Initialize the GAT layer.
        :param attn_heads: Number of attention heads.
        :param hid_units: Number of hidden units per attention head.
        :param activation: Activation function.
        :param residual: Whether to use residual connection.
        :param dropout_rate: Dropout rate.
        :param use_bias: Whether to use bias.
        """
        super(SpatioTemporalGAT, self).__init__()
        self.attn_heads = attn_heads
        self.hid_units = hid_units
        self.activation = activation
        self.residual = residual
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias

        # Dropout layer
        self.dropout = Dropout(self.dropout_rate)

    def build(self, input_shape):

        feature_dim = input_shape[-1]

        # Linear transformation weights (for feature transformation)
        self.W = self.add_weight(
            shape=(feature_dim, self.hid_units * self.attn_heads),
            initializer="glorot_uniform",  # Xavier initialization
            name="W",
        )

        # Attention mechanism weights (for computing attention scores)
        self.a = self.add_weight(shape=(2 * self.hid_units,), initializer="glorot_uniform", name="a")

        # Bias (optional)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.hid_units * self.attn_heads,), initializer="zeros", name="bias")

    def call(self, inputs, adj_matrix):
        """
        Forward pass.
        :param inputs: Input tensor [B, T, C, F].
        :param adj_matrix: Adjacency matrix [B, T*C, T*C, 1].
        :return: Output tensor [B, T, C, attn_heads * hid_units].
        """
        B, T, C, F = tf.shape(inputs)

        # Linear transformation
        h = tf.matmul(tf.reshape(inputs, [B, T * C, F]), self.W)  # [B, T*C, hid_units * attn_heads]
        h = self.dropout(h)
        h = tf.reshape(h, [B, T * C, self.attn_heads, self.hid_units])  # [B, T*C, attn_heads, hid_units]

        # Compute attention scores
        h_i = tf.tile(tf.expand_dims(h, 2), [1, 1, T * C, 1, 1])  # [B, T*C, T*C, attn_heads, hid_units]
        h_j = tf.tile(tf.expand_dims(h, 1), [1, T * C, 1, 1, 1])  # [B, T*C, T*C, attn_heads, hid_units]
        concat_h = tf.concat([h_i, h_j], axis=-1)  # [B, T*C, T*C, attn_heads, 2 * hid_units]
        e = tf.nn.leaky_relu(tf.reduce_sum(concat_h * self.a, axis=-1))  # [B, T*C, T*C, attn_heads]
        e = tf.where(adj_matrix > 0, e, tf.constant(-1e9, dtype=e.dtype))  # Apply adjacency matrix mask

        # Normalize attention coefficients
        alpha = tf.nn.softmax(e, axis=2)  # [B, T*C, T*C, attn_heads]
        alpha = self.dropout(alpha)

        # Feature aggregation
        h_prime = tf.reduce_sum(tf.expand_dims(alpha, -1) * h_j, axis=2)  # [B, T*C, attn_heads, hid_units]
        h_prime = tf.reshape(h_prime, [B, T, C, self.attn_heads * self.hid_units])  # [B, T, C, attn_heads * hid_units]

        # Add bias
        if self.use_bias:
            h_prime += self.bias

        # Residual connection
        if self.residual:
            h_prime += inputs

        return self.activation(h_prime)

    def get_config(self):
        """
        Get the configuration of the layer.
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


def set_dynamic_matrix(static_matrix, feature_shape):

    B, T, C, _ = feature_shape

    # 1. Generate spatio-temporal adjacency matrix [T*C, T*C]
    horizontal_tiled = tf.tile(static_matrix, [1, T])  # [C, T*C]
    spatio_temporal_adj = tf.tile(horizontal_tiled, [T, 1])  # [T*C, T*C]

    # 2. Generate identity matrix [T*C, T*C]
    identity_matrix = tf.eye(T * C, dtype=tf.float32)

    # 3. Combine spatio-temporal adjacency matrix with identity matrix
    dynamic_matrix = spatio_temporal_adj + identity_matrix
    dynamic_matrix = dynamic_matrix[tf.newaxis, :, :, tf.newaxis]  # [1, T*C, T*C, 1]
    dynamic_matrix = tf.tile(dynamic_matrix, [B, 1, 1, 1])  # [B, T*C, T*C, 1]

    return dynamic_matrix


if __name__ == "__main__":
    B, T, C, F = 2, 3, 4, 5
    inputs = tf.random.normal([B, T, C, F])
    static_adj_matrix = tf.constant([[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]], dtype=tf.float32)
    dynamic_matrix = set_dynamic_matrix(static_adj_matrix, inputs.shape)

    gat_layer = SpatioTemporalGAT(attn_heads=2, hid_units=16, residual=False, dropout_rate=0.2)

    # 前向传播
    output = gat_layer(inputs, dynamic_matrix)
    print("输入形状:", inputs.shape)
    print("输出形状:", output.shape)
