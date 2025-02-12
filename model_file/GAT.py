# -*- coding: utf-8 -*-
# @Time : 2024/1/11 23:20
# @Author : Yuxin Zhao
# @File : GAT.py
# @Software: Vscode

import tensorflow as tf


class GAT(tf.keras.layers.Layer):
    def __init__(self,attn_heads, hid_units, activation=tf.nn.elu, residual=False,dropout_rate_in=0.2,dropout_rate_out=0.2):
        super(GAT, self).__init__()
        
        self.attn_heads = attn_heads
        self.hid_units = hid_units
        self.activation = activation
        self.residual = residual
        self.dropout_rate_in = dropout_rate_in
        self.dropout_rate_out = dropout_rate_out
        self.dropout_in = tf.keras.layers.Dropout(rate=self.dropout_rate_in)
        self.dropout_out = tf.keras.layers.Dropout(rate=self.dropout_rate_out)
        self.batch_norm = tf.keras.layers.BatchNormalization()
        
    def build(self,input_shape):
        self.W_1 = self.add_weight(shape=(self.hid_units[0], 5),initializer='he_normal',trainable=True,name='W_1')
        self.Z_1 = self.add_weight(shape=(self.hid_units[0], 1),initializer='he_normal',trainable=True,name='Z_1')
        self.W_2 = self.add_weight(shape=(1, 5),initializer='he_normal',trainable=True,name='W_2')
        self.Z_2 = self.add_weight(shape=(1, 64),initializer='he_normal',trainable=True,name='Z_2')
    def attn_head(self,input,bias_mat):
        

        seq_fts= self.Z_1 + tf.matmul(self.W_1,input) 
        f_1 = self.Z_2 + tf.matmul(self.W_2 ,input) 
        f_2 = self.Z_2 + tf.matmul(self.W_2 ,input)
        logits = tf.transpose(f_1,[0, 2, 1]) + f_2
        coefs = tf.nn.softmax(tf.nn.leaky_relu(logits) + bias_mat)
        coefs = self.dropout_out(coefs)
        seq_fts = self.dropout_in(seq_fts)
        vals = tf.matmul(seq_fts,coefs)
        return self.activation(vals)
    def call(self, inputs,bias_mat):
        attns = []
        for _ in range(self.attn_heads[0]):
            out_1=self.attn_head(input=inputs,bias_mat=bias_mat)
            attns.append(out_1)
        h_1 = tf.concat(attns, axis=-2) 
        h_1 = self.batch_norm(h_1)
        return h_1
    def get_config(self):

        config = super().get_config()

        config.update({
            "attn_heads": self.attn_heads,
            "hid_units": self.hid_units,
            "activation": self.activation,
            "residual": self.residual,
            "dropout_rate_in": self.dropout_rate_in,
            "dropout_rate_out":self.dropout_rate_out
        })

        return config

        