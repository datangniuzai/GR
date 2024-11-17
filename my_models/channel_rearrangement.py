#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2024/11/14 19:23
# @Author : 李 嘉 轩
# @File : channel_rearrangement.py
# @Software: PyCharm

import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Layer
import config as cf

@tf.keras.utils.register_keras_serializable(package="Custom", name="ChannelSelector")
class ChannelSelector(Layer):
    def __init__(self, start_index, step_size, **kwargs):
        super(ChannelSelector, self).__init__(**kwargs)
        self.start_index = start_index
        self.step_size = step_size

    def call(self, inputs):

        selected_indices = tf.range(self.start_index, tf.shape(inputs)[-2], self.step_size)
        return tf.gather(inputs, selected_indices, axis=-2)

    def get_config(self):
        config = super(ChannelSelector, self).get_config()
        config.update({
            "start_index": self.start_index,
            "step_size": self.step_size
        })
        return config

def channels_trans():
    new_column_order = [2, 4, 6, 8, 10, 1, 3, 5, 7, 9, 11, 13, 12, 14, 16, 17, 19, 21, 23, 25, 27, 29, 31, 33, 34,
                        36, 38, 40, 42, 44, 46, 48, 49, 51, 53, 55, 57, 59, 61, 63, 41, 43, 45, 47, 50, 52, 54, 56,
                        58, 60, 62, 64, 22, 20, 18, 15, 30, 28, 26, 24, 39, 37, 35, 32]
    # 使用列表推导式将每个元素减一
    new_column_order_minus_one = [x - 1 for x in new_column_order]
    for gesture_number in cf.gesture:
        path = cf.folder_path + f'output_data/sEMG_data{gesture_number}.csv'
        df = pd.read_csv(path, header=None)
        df = df.reindex(columns=new_column_order_minus_one)
        df.to_csv(cf.folder_path+f'output_data/trans_sEMG_data{gesture_number}.csv', index=False, header=False)
        print(f"{gesture_number}号数据已处理好")