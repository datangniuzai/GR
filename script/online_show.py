#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/1/28 13:40
# @Author : JIAXUAN LI
# @File : online_show.py
# @Software: PyCharm

import os
import time
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import socket
import struct
import numpy as np
import tkinter as tk
import tensorflow as tf
from tkinter import Label

import config as cf
from dataset import calc_td
from model_file import creat_model
from filtering import bandpass_and_notch_filter

def online_data_read(self, queue):
    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_socket.bind(('192.168.1.100', 8080))
    try:
        output_data = np.empty((self.window_size, 64), dtype=np.float32)
        idx = 0
        while True:
            data, addr = udp_socket.recvfrom(1300)
            transposed_data = np.frombuffer(data[18:1298], dtype='<i2').reshape(10, 64)
            output_data[idx:idx+10, :] = transposed_data
            idx += 10
            if idx == self.window_size:
                queue.put(output_data)
                idx = 0
    finally:
        udp_socket.close()

def online_td_calc(gestures ,self, queue,model_path):
    cf.model = creat_model()
    cf.model.load_weights(model_path)
    while True:
        if not queue.empty():
            data = queue.get() /self.scaling
            filtered_data = bandpass_and_notch_filter(data)
            window_data_feature = tf.convert_to_tensor(calc_td(filtered_data), dtype=tf.float32)
            predictions = cf.model.predict(window_data_feature)
            predicted_class = np.argmax(predictions, axis=1)
            predicted_gesture = [gestures[i] for i in predicted_class]
            print(f"Predicted gesture: NO.{predicted_gesture} gesture")
        time.sleep(0.1)


if __name__ == '__main__':
    cf.config_read()
    cf.model = creat_model()




