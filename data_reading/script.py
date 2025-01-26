#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2024/11/14 19:43
# @Author : 李 嘉 轩
# @File : script.py
# @Software: PyCharm

import os
import json
import time
import socket
import struct
import pyttsx3
import datetime
import numpy as np
import tensorflow as tf

import config as cf

def sEMG_data_read_save():

    engine = pyttsx3.init()
    rate = engine.getProperty('rate')
    engine.setProperty('rate', rate + 50)  # 修改数字以调整语速，数字越大语速越快
    # 选择不同的声音（如果可用）
    # voices = engine.getProperty('voices')
    # engine.setProperty('voice', voices[1].id)  # 选择第二个声音（索引从0开始）

    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_socket.bind(('192.168.1.100', cf.collector_number))
    start_time = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
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
                        with open(cf.data_path + f'original_data/sEMG_data{gesture_number}.csv', 'a') as f:
                            np.savetxt(f, output_data[cf.sample_rate:, :] * 0.195, delimiter=',', fmt='%.6f')
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

        end_time = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

        generate_volunteer_experiment_info(start_time,end_time)

        udp_socket.close()

        print(f"❤️Please rename the folder [{cf.data_path}] to identifier "
              "and complete the details of the [vol_exp_info.json].")

def _parse_function(proto):
    """
    解析 TFRecord 文件中的每个 Example，并对数据类型和形状进行调整
    :param proto: 输入的 TFRecord 数据
    :return: 解析后的字典，包括窗口数据、标签和其他特征
    """
    keys_to_features = {
        'window': tf.io.FixedLenFeature(cf.feature_shape, tf.float32),
        'adjacency': tf.io.FixedLenFeature([64,64], tf.float32),
        'label': tf.io.FixedLenFeature([1], tf.int64),
        'time_preread_index': tf.io.FixedLenFeature([1], tf.int64),
        'window_index': tf.io.FixedLenFeature([1], tf.int64)
    }

    data = tf.io.parse_single_example(proto, keys_to_features)
    # retype
    data['label'] = tf.cast(data['label'], tf.uint8)
    data['time_preread_index'] = tf.cast(data['time_preread_index'], tf.uint8)
    data['window_index'] = tf.cast(data['window_index'], tf.uint8)

    return data["window"],data['adjacency'], data['label'], data['time_preread_index'], data['window_index']

def generate_volunteer_experiment_info(start_time,end_time):

    experiment_info = {
        "name": "volunteer_experiment_info",
        "description": "Details of subjects and experimental process",
        "detailed description": "",
        "note": "The following description of time is in hours.",
        "explanation about identifier": "date/subject's_last_name/man_or_female/static_or_dynamic/number_of_gestures",
        "identifier": "250112-Z-Man-S-26",
        "volunteer_info": {
            "name": "",
            "age": "",
            "gender": "male/female",
            "measured_arm": "left/right",
            "diet": "Normal",
            "weekly_exercise_duration": 3.5,
            "subject_conditions": {
                "neurological_diseases": "None",
                "physical_conditions": "Healthy",
                "sleep": {
                    "previous_night_sleep_duration": 7.5,
                    "bedtime": "2024-11-17T23:00:00"
                },
                "diet": "Normal",
                "weekly_exercise_duration": 3.5
            }
        },
        "experiment_info": {
            "gesture_sequence": cf.gesture,
            "collector_number": cf.collector_number - 8079,
            "gesture_read_count_per_instance":cf.turn_read_sum,
            "read_duration_per_instance": cf.time_preread,
            "experiment_time": {
                "start_time": start_time,
                "end_time": end_time,
                "gesture_rest": cf.gesture_rest,
                "action_rest":cf.action_rest
            }
        }
    }
    file_name = os.path.join(cf.data_path, "vol_exp_info.json")
    with open(file_name, 'w', encoding='utf-8') as f:
        json.dump(experiment_info, f, ensure_ascii=False, indent=4)

    print(f"实验记录数据已保存至: {file_name}")

def load_tfrecord(tfrecord_path):
    """
    从 TFRecord 文件中加载数据
    :param tfrecord_path: TFRecord 文件的路径
    :return: 返回一个包含窗口数据、标签、时间先读索引和窗口索引的 TensorFlow 数据集
    """
    # 创建 TFRecord 数据集
    dataset = tf.data.TFRecordDataset(tfrecord_path)

    # 使用 `_parse_function` 解析 TFRecord 中的数据
    dataset = dataset.map(_parse_function)

    return dataset

def load_tfrecord_list(tfrecord_path):
    """
    加载 TFRecord 文件并返回一个数据集
    :param tfrecord_path: TFRecord 文件路径
    :return: 一个包含窗口数据、标签和其他特征的 TensorFlow 数据集
    """

    dataset = tf.data.TFRecordDataset(tfrecord_path)

    dataset = dataset.map(_parse_function)

    window_datas = []
    adjacencys = []
    labels = []
    time_preread_indices = []
    window_indices = []

    for window_data,adjacency,label, time_preread_index, window_index in dataset:
        window_datas.append(window_data.numpy())
        adjacencys.append(adjacency.numpy())
        labels.append(label.numpy())
        time_preread_indices.append(time_preread_index.numpy())
        window_indices.append(window_index.numpy())
    
    return window_datas,adjacencys,labels, time_preread_indices, window_indices

