#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2024/11/14 19:48
# @Author : 李 嘉 轩
# @File : script.py
# @Software: PyCharm


import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf

import config as cf
from data_reading import load_tfrecord
from filtering import bandpass_and_notch_filter
from dataset.build_adjacency import build_one_adjacency, adj_to_bias, fill_new_adjacency_matrix, sample_neighbors
from dataset.caculate_features import zero_crossing_rate, mean_absolute_value, calculate_mean_frequency, calculate_median_frequency, WL, root_mean_square

def z_score_normalize(data):
    mean_vals = np.mean(data, axis=-2, keepdims=True)
    std_vals = np.std(data, axis=-2, keepdims=True)
    normalized_data = (data - mean_vals) / (std_vals+0.000001)
    return normalized_data

def calc_TD(data: np.ndarray) -> np.ndarray:
    """
    Extract only time-domain features, add small windows. Feature order: MAV, RMS, MSE, Zero-crossings, WAMP.

    :param data: Input data matrix with shape (num_channels, signal_length)
    :return: Extracted features with shape (num_windows, num_channels, 5)
    """
    num_channels, signal_length = data.shape
    num_windows = (signal_length - cf.window_size_little) // cf.step_size_little + 1
    features = []

    for i in range(num_windows):

        start = i * cf.step_size_little
        end = start + cf.window_size_little
        windowed_data = data[:, start:end]

        max_possible_rms = np.max(windowed_data) / 2
        max_possible_mse = (np.max(windowed_data) / 2) ** 2
        max_possible_zero_crossings = cf.step_size_little - 1
        max_possible_willison_amplitudes = cf.step_size_little - 1

        # Mean Absolute Value (MAV)
        mav = np.mean(np.abs(windowed_data), axis=1)
        # Root Mean Square (RMS)
        rms = np.sqrt(np.mean(windowed_data ** 2, axis=1)) / max_possible_rms
        # Mean Squared Error (MSE)
        mse = np.mean((windowed_data - np.sqrt(np.mean(windowed_data ** 2, axis=0))) ** 2, axis=1) / max_possible_mse

        # Zero-crossings
        zero_crossings = []
        for channel in windowed_data:
            crossings = (np.sum(np.diff(np.sign(channel)) != 0)) / cf.step_size_little
            zero_crossings.append(crossings / max_possible_zero_crossings)

        # Willison Amplitude (WAMP)
        threshold = 20 / cf.scaling
        willison_amplitudes = []
        for channel in windowed_data:
            differences = np.diff(channel)
            willison_amplitude = np.sum(np.abs(differences) > threshold)
            willison_amplitudes.append(willison_amplitude / max_possible_willison_amplitudes)

        # Stack the features together
        feature = np.array([mav, rms, mse, np.array(zero_crossings), np.array(willison_amplitudes)])
        features.append(feature.T)

    features = np.array(features)
    z_score_normalize(features)
    return features

def generate_smallwindowed_data(data, window_size,stride):
    num_channels, signal_length = data.shape
    num_windows = (signal_length - window_size) //stride + 1
    windowed_data = np.zeros((num_windows,num_channels,window_size))
    for i in range(num_windows):
        start = i * stride
        end = start + window_size
        windowed_data[i,:,:] = data[:, start:end]
    return windowed_data

def time_features(data):
    data = generate_smallwindowed_data(data,cf.window_size_little,cf.step_size_little)
    feature_functions = [
        zero_crossing_rate,mean_absolute_value,calculate_mean_frequency,calculate_median_frequency,WL,
        root_mean_square]
    features = np.zeros((data.shape[0],data.shape[1],len(feature_functions)))
    data = [feature_func(data) for feature_func in feature_functions]
    for i in range(len(feature_functions)):
        features[:,:,i]=data[i]
    features = z_score_normalize(features)
    return features

def online_dataset():
    print("正在处理数据，请稍等")
    print(
        f"取第{cf.train_nums}次采集的数据作为训练集，\n"
        f"取第{cf.test_nums}次采集的数据作为测试集，\n"
        f"取第{cf.val_nums}次采集的数据作为验证集\n"
    )

    for gesture_number in cf.gesture:
        path = cf.data_path + f'original_data/sEMG_data{gesture_number}.csv'
        df = pd.read_csv(path, header=None).to_numpy().T
        for dataset_type in ['train','val','test']:
            dataset_establish(df, gesture_number, dataset_type)
        print(f"{gesture_number}号手势数据处理完毕")

def dataset_establish(df: np.ndarray, gesture_number:int, dataset_type :str):
    """
    通用数据处理函数，用于训练集、测试集和验证集的特征提取和保存
    :param df: 输入信号
    :param gesture_number: 手势编号
    :param dataset_type: 数据集类型（'train'/'test'/'val'）
    :return: 数据集的element_spec
    """
    window_data_feature = []
    window_data_label = []

    for read_time in range(1, cf.turn_read_sum + 1):
        if read_time in getattr(cf, f"{dataset_type}_nums"):
            single_acqui_data = df[:,(read_time - 1) * (cf.time_preread * cf.sample_rate):read_time * (cf.time_preread * cf.sample_rate)]
            for j in range(0, single_acqui_data.shape[1] - cf.window_size + 1, cf.step_size):
                window_data = single_acqui_data[:, j:j + cf.window_size]
                window_data = bandpass_and_notch_filter(window_data)
                window_data_feature.append(calc_TD(window_data))
                window_data_label.append(gesture_number - 1)

    graph_nums = len(window_data_feature)
    adjacency = adj_to_bias(fill_new_adjacency_matrix(sample_neighbors(np.array([build_one_adjacency()] * graph_nums),5),graph_nums,64))

    window_data_feature_tensor = tf.convert_to_tensor(window_data_feature, dtype=tf.float32)
    adjacency_tensor = tf.convert_to_tensor(adjacency,dtype=tf.float32)
    label_tensor = tf.convert_to_tensor(window_data_label, dtype=tf.uint8)

    dataset = tf.data.Dataset.from_tensor_slices((window_data_feature_tensor,adjacency_tensor,label_tensor))

    save_path = os.path.join(cf.data_path,"processed_data")
    os.makedirs(save_path, exist_ok=True)

    tfrecord_path = os.path.join(save_path, f"data_{gesture_number}_{dataset_type}.tfrecord")

    save_tfrecord(dataset,tfrecord_path)

def dataset_connect():

    for dataset_type in ['train','test','val']:

        merged_dataset = None

        for gesture_number in cf.gesture:

            dataset= load_tfrecord(cf.data_path + f"processed_data/data_{gesture_number}_{dataset_type}.tfrecord")

            if merged_dataset is None:
                merged_dataset = dataset
            else:
                merged_dataset = merged_dataset.concatenate(dataset)

        connect_tfrecord_save_path = os.path.join(cf.data_path,f"processed_data/data_contact_{dataset_type}.tfrecord")

        save_tfrecord(merged_dataset,connect_tfrecord_save_path)

        print(f"[{dataset_type}]数据已合并并保存在[{connect_tfrecord_save_path}]")

def save_tfrecord(dataset :tf.data.Dataset,tfrecord_save_path:str):
    """
    将数据集保存为 TFRecord 文件。

    参数：
    dataset (iterable)：包含窗口数据、标签、生成的邻接矩阵。
    tfrecord_save_path (str)：保存 TFRecord 文件的路径。

    功能：
    将数据集中的每个项（窗口数据、标签等）转换为 `tf.train.Example` 格式并写入指定的 TFRecord 文件。
    """
    with tf.io.TFRecordWriter(tfrecord_save_path) as writer:
        for window, adjacency, label in dataset:
            feature = {
                'window': tf.train.Feature(float_list=tf.train.FloatList(value=window.numpy().flatten())),
                'adjacency': tf.train.Feature(float_list=tf.train.FloatList(value=adjacency.numpy().flatten())),
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label.numpy()])),
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())





