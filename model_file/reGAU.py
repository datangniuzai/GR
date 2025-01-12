#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2024/11/14 19:48
# @Author : 李 嘉 轩
# @team member : 赵雨新
# @File : script.py
# @Software: PyCharm Vscode

import tensorflow as tf
from model_file.GAT import GAT
class reGAU(tf.keras.layers.Layer):
    def __init__(self,in_channels,out_channels,attn_heads,hid_units,dropout_rate_in = 0.2,dropout_rate_out=0.3,**kwargs):
        super(reGAU, self).__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.attn_heads = attn_heads
        self.hid_units = hid_units
        self.dropout_rate_in = dropout_rate_in
        self.dropout_rate_out = dropout_rate_out
    
        self._create_parameters_and_layers()
    def build(self,input_shape):
        self.W_z = self.add_weight(
            shape=(self.in_channels, self.out_channels),
            initializer='he_normal',
            trainable=True,
            name='W_z'
        )

        self.Z_bias = self.add_weight(
            shape=(1, self.out_channels),
            initializer=tf.keras.initializers.HeNormal(),
            trainable=True,
            name='Z_bias'
        )
        self.W_h = self.add_weight(
            shape=(self.in_channels, self.out_channels),
            initializer='he_normal',
            trainable=True,
            name='W_h'
        )
        self.H_bias = self.add_weight(
            shape=(1, self.out_channels),
            initializer=tf.keras.initializers.HeNormal(),
            trainable=True,
            name='H_bais'
        )

        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.dropout_in = tf.keras.layers.Dropout(rate=self.dropout_rate_in)
        self.dropout_out = tf.keras.layers.Dropout(rate=self.dropout_rate_out)
    def _calculate_update_gate_parameters_and_layers(self):
        self.conv_z = GAT(self.attn_heads, self.hid_units)
    def _calculate_hidden_gate_parameters_and_layers(self):
        self.conv_h = GAT(self.attn_heads, self.hid_units)
    def _create_parameters_and_layers(self):
        self._calculate_update_gate_parameters_and_layers()
        self._calculate_hidden_gate_parameters_and_layers()
    def _set_hidden_state(self, X, H):
        if H is None:
            batch_size = tf.shape(X)[0]
            shape = (batch_size, X.shape[2], self.out_channels)
            H = tf.zeros(shape, dtype=tf.float32)
        return H

    def _calculate_update_gate(self, X, edge_index, H):
        Z = self.conv_z(inputs=X, bias_mat=edge_index)
        Z = Z +tf.matmul(X,self.W_z)+self.Z_bias+H
        Z = tf.sigmoid(Z)
        return Z
    def _calculate_hidden_state(self, X, edge_index, H, Z):
        T = self.conv_h(inputs=X, bias_mat=edge_index)
        T = T + H +tf.matmul(X,self.W_h)+self.H_bias
        T = tf.tanh(T)
        H = Z * H +(1-Z)*T
        return H

    def call(self,edge_index, X, H=None, last_layer=True):
        H = self._set_hidden_state(X, H)
        out_puts = []

        for i in range(X.shape[1]):
            Z = self._calculate_update_gate(X[:, i, :, :], edge_index, H)
            H = self._calculate_hidden_state(X[:, i, :, :],edge_index, H, Z)
            H = self.dropout_out(H)
            outdata = H
            outdata = tf.expand_dims(outdata, axis=-1)
            out_puts.append(outdata)

        out_puts = tf.concat(out_puts, axis=-1)
        if last_layer:
            out_puts = H
            H = self.batch_norm(out_puts)
        else:
            out_puts = tf.transpose(out_puts, [0, 3, 1, 2])
            H = self.batch_norm(out_puts)
        H = self.dropout_out(H)
        return H

    def get_config(self):

        config = super().get_config()

        config.update({
            "in_channels":self.in_channels,
            "out_channels":self.out_channels,
            "attn_heads": self.attn_heads,
            "hid_units": self.hid_units,
            "dropout_rate_in": self.dropout_rate_in,
            "dropout_rate_out":self.dropout_rate_out
        })
        return config
