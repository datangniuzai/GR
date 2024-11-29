#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2024/11/14 19:48
# @Author : 李 嘉 轩
# @File : script.py
# @Software: PyCharm

import os
import numpy as np
import pandas as pd
import tensorflow as tf

import config as cf
from filtering import bandpass_and_notch_filter
from data_reading import load_tfrecord

def time_features(data):
    """
    只提取时域特征，加小窗，特征顺序：MAV，RMS，MSE，过零点数，WAMP
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

        mav = np.mean(np.abs(windowed_data), axis=1)
        rms = np.sqrt(np.mean(windowed_data ** 2, axis=1)) / max_possible_rms
        mse = np.mean((windowed_data - np.sqrt(np.mean(windowed_data ** 2, axis=0))) ** 2, axis=1) / max_possible_mse

        # 过零点数
        zero_crossings = []
        for channel in windowed_data:
            crossings = (np.sum(np.diff(np.sign(channel)) != 0)) / cf.step_size_little
            zero_crossings.append(crossings / max_possible_zero_crossings)

        # Willison幅值
        threshold = 20 / cf.scaling
        willison_amplitudes = []
        for channel in windowed_data:
            differences = np.diff(channel)
            willison_amplitude = np.sum(np.abs(differences) > threshold)
            willison_amplitudes.append(willison_amplitude / max_possible_willison_amplitudes)
        feature = np.array([mav, rms, mse, np.array(zero_crossings), np.array(willison_amplitudes)])
        features.append(feature.T)
    features = np.array(features)
    return features

def time_frequency_features(data):
    """
    提取时频域特征，特征顺序：MAV，RMS，MSE，过零点数，Willison幅值，中值频率（Median Frequency），均值频率（Mean Frequency），频率比（Frequency Ratio）
    """
    len_data = data.shape[1]
    max_possible_rms = np.max(data) / 2
    max_possible_mse = (np.max(data) / 2) ** 2
    max_possible_zero_crossings = len_data - 1
    max_possible_willison_amplitudes = len_data - 1
    nyquist_frequency = cf.sample_rate / 2

    mav = np.mean(np.abs(data), axis=1)
    rms = np.sqrt(np.mean(data ** 2, axis=1)) / max_possible_rms
    mse = np.mean((data - np.sqrt(np.mean(data ** 2, axis=0))) ** 2, axis=1) / max_possible_mse

    # 过零点数
    zero_crossings = []
    for channel in data:
        crossings = (np.sum(np.diff(np.sign(channel)) != 0)) / len_data
        zero_crossings.append(crossings / max_possible_zero_crossings)

    # Willison幅值
    threshold = 20 / cf.scaling
    willison_amplitudes = []
    for channel in data:
        differences = np.diff(channel)
        willison_amplitude = np.sum(np.abs(differences) > threshold)
        willison_amplitudes.append(willison_amplitude / max_possible_willison_amplitudes)
    # 应用汉宁窗
    window = np.hanning(cf.window_size_little)
    windowed_data_han = data * window
    # 频率特征
    fft_results = np.zeros((windowed_data_han.shape[0], len_data // 2), dtype=complex)
    for i in range(windowed_data_han.shape[0]):
        fft_result = np.fft.fft(windowed_data_han[i])
        fft_results[i] = fft_result[:len_data // 2]  # 只取正频率部分

    frequencies = np.fft.fftfreq(len_data, 1 / cf.sample_rate)[:windowed_data_han.shape[1] // 2]
    mdf_values = []
    mnf_values = []
    fr_values = []
    for i in range(windowed_data_han.shape[0]):
        magnitude_spectrum = np.abs(fft_results[i])
        non_zero_indices = magnitude_spectrum > 0
        cumulative_magnitude = np.cumsum(magnitude_spectrum[non_zero_indices])
        half_total_magnitude = cumulative_magnitude[-1] / 2
        mdf_index = np.argmax(cumulative_magnitude >= half_total_magnitude)
        mdf = frequencies[non_zero_indices][mdf_index] / nyquist_frequency
        mdf_values.append(mdf)

        mnf = np.average(frequencies, weights=magnitude_spectrum) / nyquist_frequency
        mnf_values.append(mnf)

        low_freq_threshold = 80
        high_freq_threshold = 160
        low_freq_energy = np.sum(magnitude_spectrum[frequencies <= low_freq_threshold])
        high_freq_energy = np.sum(magnitude_spectrum[frequencies >= high_freq_threshold])
        fr = high_freq_energy / low_freq_energy if low_freq_energy != 0 else float('inf')
        fr_values.append(fr)
    feature = (
        np.array([mav, rms, mse, np.array(zero_crossings), np.array(willison_amplitudes), np.array(mdf_values),
                  np.array(mnf_values), np.array(fr_values)]))
    return feature

def online_dataset():
    print("正在处理数据，请稍等")
    print(
        f"取第{cf.train_nums}次采集的数据作为训练集，\n"
        f"取第{cf.test_nums}次采集的数据作为测试集，\n"
        f"取第{cf.val_nums}次采集的数据作为验证集\n"
    )

    for gesture_number in cf.gesture:
        path = cf.data_path + f'original_data/sEMG_data{gesture_number}.csv'
        df = pd.read_csv(path, header=None).to_numpy().T / cf.scaling

        for dataset_type in ['train', 'test', 'val']:
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
    window_data_time_preread_index = []
    window_data_window_index = []

    for i in range(cf.turn_read_sum):
        if i in getattr(cf, f"{dataset_type}_nums"):
            data = df[:, i * (cf.time_preread * cf.sample_rate):
                         (i + 1) * (cf.time_preread * cf.sample_rate)]
            for j in range(0, data.shape[1] - cf.window_size + 1, cf.step_size):
                window_data = data[:, j:j + cf.window_size]
                window_data = bandpass_and_notch_filter(window_data)
                window_data_feature.append(time_features(window_data))
                window_data_label.append(gesture_number)
                window_data_time_preread_index.append(i)
                window_data_window_index.append(j)

    window_data_feature_tensor = tf.convert_to_tensor(window_data_feature, dtype=tf.float32)
    label_tensor = tf.convert_to_tensor(window_data_label, dtype=tf.uint8)
    time_preread_index_tensor = tf.convert_to_tensor(window_data_time_preread_index, dtype=tf.uint8)
    window_index_tensor = tf.convert_to_tensor(window_data_window_index, dtype=tf.uint8)
    dataset = tf.data.Dataset.from_tensor_slices((window_data_feature_tensor, label_tensor,
                                                  time_preread_index_tensor, window_index_tensor))
    save_path = os.path.join(cf.data_path,"processed_data")
    os.makedirs(save_path, exist_ok=True)
    tfrecord_path = os.path.join(save_path, f"data_{gesture_number}_{dataset_type}.tfrecord")
    with tf.io.TFRecordWriter(tfrecord_path) as writer:
        for window, label, time_preread_index, window_index in dataset:
            feature = {
                'window': tf.train.Feature(float_list=tf.train.FloatList(value=window.numpy().flatten())),
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label.numpy()])),
                'time_preread_index': tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[time_preread_index.numpy()])),
                'window_index': tf.train.Feature(int64_list=tf.train.Int64List(value=[window_index.numpy()])),
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())

def dataset_connect():

    for dataset_type in ['train', 'test', 'val']:

        merged_dataset = None

        for gesture_number in cf.gesture:

            dataset= load_tfrecord(cf.data_path + f"processed_data/data_{gesture_number}_{dataset_type}.tfrecord")

            if merged_dataset is None:
                merged_dataset = dataset
            else:
                merged_dataset = merged_dataset.concatenate(dataset)

        tfrecord_save_path = os.path.join(cf.data_path,f"processed_data/data_contact_{dataset_type}.tfrecord")
        save_tfrecord(merged_dataset,tfrecord_save_path)
        print(f"[{dataset_type}]数据已合并并保存在[{tfrecord_save_path}]")

def save_tfrecord(dataset :tf.data.Dataset,tfrecord_save_path:str):
    """
    将数据集保存为 TFRecord 文件。

    参数：
    dataset (iterable)：包含窗口数据、标签、时间索引和窗口索引的迭代器。
    tfrecord_save_path (str)：保存 TFRecord 文件的路径。

    功能：
    将数据集中的每个项（窗口数据、标签等）转换为 `tf.train.Example` 格式并写入指定的 TFRecord 文件。
    """
    with tf.io.TFRecordWriter(tfrecord_save_path) as writer:
        for window, label, time_preread_index, window_index in dataset:
            feature = {
                'window': tf.train.Feature(
                    float_list=tf.train.FloatList(value=window.numpy().flatten())),
                'label': tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[label.numpy().item()])),
                'time_preread_index': tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[time_preread_index.numpy().item()])),
                'window_index': tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[window_index.numpy().item()])),
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())






