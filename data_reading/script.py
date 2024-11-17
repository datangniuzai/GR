#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2024/11/14 19:43
# @Author : 李 嘉 轩
# @File : script.py
# @Software: PyCharm

import os
import yaml
import time
import socket
import struct
import pyttsx3
import numpy as np
import tensorflow as tf
import config as cf

def online_data_read():

    engine = pyttsx3.init()
    rate = engine.getProperty('rate')
    engine.setProperty('rate', rate + 50)  # 修改数字以调整语速，数字越大语速越快
    # 选择不同的声音（如果可用）
    # voices = engine.getProperty('voices')
    # engine.setProperty('voice', voices[1].id)  # 选择第二个声音（索引从0开始）

    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_socket.bind(('192.168.1.100', 8080))

    try:
        output_data = np.empty((0, 64), dtype=np.float32)
        i = 1
        while i < (cf.turn_read_sum+1):
            for gesture_number in cf.gesture:
                text_to_speak = str(f"请做好{gesture_number}号手势,采集开始")
                print(text_to_speak)
                engine.say(text_to_speak)
                engine.runAndWait()
                time.sleep(0.5)
                print("开始采集")
                turn = True
                while turn:
                    data, addr = udp_socket.recvfrom(1300)
                    reshaped_data = np.reshape(np.array(struct.unpack('<640h', data[18:1298])), (10, 64))
                    output_data = np.concatenate((output_data, reshaped_data), axis=0)
                    if output_data.shape[0] == (cf.time_preread + 1) * cf.sample_rate:
                        with open(cf.folder_path + f'output_data/sEMG_data{gesture_number}.csv', 'a') as f:
                            np.savetxt(f, output_data[cf.sample_rate:, :] * 0.195, delimiter=',', fmt='%.6f')
                            # 在这里加一个数据可视化，“子图”，pyplot
                        turn = False
                        output_data = np.empty((0, 64), dtype=np.float32)
                time.sleep(0.5)
                text_to_speak = str(f"请休息")
                print(text_to_speak)
                engine.say(text_to_speak)
                engine.runAndWait()
                time.sleep(15)
            i += 1
            time.sleep(180)

    finally:

        udp_socket.close()

def read_tfrecord(file_path):
    # 从TFRecord文件中读取数据并解析
    raw_dataset = tf.data.TFRecordDataset(file_path)
    parsed_dataset = raw_dataset.map(parse_example)
    # 遍历解析后的数据集并返回窗口数据、标签、time_preread_index 和 window_index
    window_data = []
    labels = []
    time_preread_indices = []
    window_indices = []
    for window, label, time_preread_index, window_index in parsed_dataset:
        window_data.append(window.numpy())
        labels.append(label.numpy())
        time_preread_indices.append(time_preread_index.numpy())
        window_indices.append(window_index.numpy())
    return window_data, labels, time_preread_indices, window_indices

def parse_example(example_proto):
    feature_description = {
        'window': tf.io.FixedLenFeature(cf.feature_shape, tf.float32),
        'label': tf.io.FixedLenFeature([], tf.int64),
        'time_preread_index': tf.io.FixedLenFeature([], tf.int64),
        'window_index': tf.io.FixedLenFeature([], tf.int64)
    }
    example = tf.io.parse_single_example(example_proto, feature_description)
    return example['window'], example['label'], example['time_preread_index'], example['window_index']

def read_volunteer_info(folder_path):
    info_file_path = os.path.join(folder_path, 'vol&ges_info.yaml')
    with open(info_file_path, 'r') as file:
        volunteer_info = yaml.safe_load(file)
    print("志愿者信息:")
    for key, value in volunteer_info.items():
        print(f"{key}: {value}")
    return volunteer_info