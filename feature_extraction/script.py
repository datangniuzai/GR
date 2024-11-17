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

import config
from data_reading import read_tfrecord
from filtering import bandpass_and_notch_filter
import config as cf

def time_features(data):
    """
    只提取时域特征，加小窗，特征顺序：MAV，RMS，MSE，过零点数，Willison幅值
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

        MAV = np.mean(np.abs(windowed_data), axis=1)
        RMS = np.sqrt(np.mean(windowed_data ** 2, axis=1)) / max_possible_rms
        MSE = np.mean((windowed_data - np.sqrt(np.mean(windowed_data ** 2, axis=0))) ** 2, axis=1) / max_possible_mse

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
        feature = np.array([MAV, RMS, MSE, np.array(zero_crossings), np.array(willison_amplitudes)])
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

    MAV = np.mean(np.abs(data), axis=1)
    RMS = np.sqrt(np.mean(data ** 2, axis=1)) / max_possible_rms
    MSE = np.mean((data - np.sqrt(np.mean(data ** 2, axis=0))) ** 2, axis=1) / max_possible_mse

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
        np.array([MAV, RMS, MSE, np.array(zero_crossings), np.array(willison_amplitudes), np.array(mdf_values),
                  np.array(mnf_values), np.array(fr_values)]))
    return feature

def feature_tract_save(data):
    """
    只提取时域特征，加小窗，特征顺序：MAV，RMS，MSE，过零点数，Willison幅值
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

        MAV = np.mean(np.abs(windowed_data), axis=1)
        RMS = np.sqrt(np.mean(windowed_data ** 2, axis=1)) / max_possible_rms
        MSE = np.mean((windowed_data - np.sqrt(np.mean(windowed_data ** 2, axis=0))) ** 2, axis=1) / max_possible_mse

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
        feature = np.array([MAV, RMS, MSE, np.array(zero_crossings), np.array(willison_amplitudes)])
        features.append(feature)
    features = np.array(features)
    return features

def online_dataset():
    print("正在处理数据，请稍等")
    print(
        f"取第{list(map(lambda x: x + 1, cf.train_nums))}次采集的数据作为训练集，"
        f"取第{list(map(lambda x: x + 1, cf.test_nums))}次采集的数据作为测试集，"
        f"取第{list(map(lambda x: x + 1, cf.val_nums))}次采集的数据作为验证集")

    for gesture_number in cf.gesture:
        path = cf.folder_path + f'output_data/sEMG_data{gesture_number}.csv'
        df = pd.read_csv(path, header=None).to_numpy().T / cf.scaling
        # 提取特征
        window_data_feature_train = []
        window_data_label_train = []
        window_data_time_preread_index_train = []
        window_data_window_index_train = []
        window_data_feature_test = []
        window_data_label_test = []
        window_data_time_preread_index_test = []
        window_data_window_index_test = []
        window_data_feature_val = []
        window_data_label_val = []
        window_data_time_preread_index_val = []
        window_data_window_index_val = []
        for i in range(cf.turn_read_sum):
            data = df[:, i * (cf.time_preread * cf.sample_rate):(i + 1) * (
                    cf.time_preread * cf.sample_rate)]
            if i in cf.test_nums:
                for j in range(0, data.shape[1] - cf.window_size + 1, cf.step_size):
                    window_data = data[:, j:j + cf.window_size]
                    window_data = bandpass_and_notch_filter(window_data)
                    window_data_feature_test.append(time_features(window_data))
                    window_data_label_test.append(gesture_number)
                    window_data_time_preread_index_test.append(i)
                    window_data_window_index_test.append(j)
            elif i in cf.val_nums:
                for j in range(0, data.shape[1] - cf.window_size + 1, cf.step_size):
                    window_data = data[:, j:j + cf.window_size]
                    window_data = bandpass_and_notch_filter(window_data)
                    window_data_feature_val.append(time_features(window_data))
                    window_data_label_val.append(gesture_number)
                    window_data_time_preread_index_val.append(i)
                    window_data_window_index_val.append(j)
            # 窗口滑动计数
            elif i in cf.train_nums:
                for j in range(0, data.shape[1] - cf.window_size + 1, cf.step_size):
                    window_data = data[:, j:j + cf.window_size]
                    window_data = bandpass_and_notch_filter(window_data)
                    window_data_feature_train.append(time_features(window_data))
                    window_data_label_train.append(gesture_number)
                    window_data_time_preread_index_train.append(i)
                    window_data_window_index_train.append(j)

        # 保存训练集为CSV文件
        train_data = np.array(window_data_feature_train)
        train_data_channel_0 = train_data[:, :, 25, :]
        train_data_channel_0_flat = train_data_channel_0.reshape(-1, train_data_channel_0.shape[-1])
        train_df = pd.DataFrame(train_data_channel_0_flat)
        train_df.to_csv(cf.folder_path + f'data/data_{gesture_number}_train_channel_0.csv', index=False)


        save_path = cf.folder_path + "data"
        # 训练集数据集建立
        window_data_feature_train_tensor = tf.convert_to_tensor(window_data_feature_train, dtype=tf.float32)
        label_train_tensor = tf.convert_to_tensor(window_data_label_train, dtype=tf.int32)
        time_preread_index_train_tensor = tf.convert_to_tensor(window_data_time_preread_index_train, dtype=tf.int32)
        window_index_train_tensor = tf.convert_to_tensor(window_data_window_index_train, dtype=tf.int32)
        dataset_train = tf.data.Dataset.from_tensor_slices((window_data_feature_train_tensor, label_train_tensor,
                                                            time_preread_index_train_tensor,
                                                            window_index_train_tensor))
        with tf.io.TFRecordWriter(os.path.join(save_path, f'data_{gesture_number}_train.tfrecord')) as writer:
            for window, label, time_preread_index, window_index in dataset_train:
                feature = {
                    'window': tf.train.Feature(float_list=tf.train.FloatList(value=window.numpy().flatten())),
                    'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label.numpy()])),
                    'time_preread_index': tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[time_preread_index.numpy()])),
                    'window_index': tf.train.Feature(int64_list=tf.train.Int64List(value=[window_index.numpy()]))
                }
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())
        # 测试集数据集建立
        window_data_feature_test_tensor = tf.convert_to_tensor(window_data_feature_test, dtype=tf.float32)
        label_test_tensor = tf.convert_to_tensor(window_data_label_test, dtype=tf.int32)
        time_preread_index_test_tensor = tf.convert_to_tensor(window_data_time_preread_index_test, dtype=tf.int32)
        window_index_test_tensor = tf.convert_to_tensor(window_data_window_index_test, dtype=tf.int32)
        dataset_test = tf.data.Dataset.from_tensor_slices((window_data_feature_test_tensor, label_test_tensor,
                                                           time_preread_index_test_tensor,
                                                           window_index_test_tensor))
        with tf.io.TFRecordWriter(os.path.join(save_path, f'data_{gesture_number}_test.tfrecord')) as writer:
            for window, label, time_preread_index, window_index in dataset_test:
                feature = {
                    'window': tf.train.Feature(float_list=tf.train.FloatList(value=window.numpy().flatten())),
                    'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label.numpy()])),
                    'time_preread_index': tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[time_preread_index.numpy()])),
                    'window_index': tf.train.Feature(int64_list=tf.train.Int64List(value=[window_index.numpy()]))
                }
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())
        # 验证集数据集建立
        window_data_feature_val_tensor = tf.convert_to_tensor(window_data_feature_val, dtype=tf.float32)
        label_val_tensor = tf.convert_to_tensor(window_data_label_val, dtype=tf.int32)
        time_preread_index_val_tensor = tf.convert_to_tensor(window_data_time_preread_index_val, dtype=tf.int32)
        window_index_val_tensor = tf.convert_to_tensor(window_data_window_index_val, dtype=tf.int32)
        dataset_val = tf.data.Dataset.from_tensor_slices((window_data_feature_val_tensor, label_val_tensor,
                                                          time_preread_index_val_tensor, window_index_val_tensor))
        with tf.io.TFRecordWriter(os.path.join(save_path, f'data_{gesture_number}_val.tfrecord')) as writer:
            for window, label, time_preread_index, window_index in dataset_val:
                feature = {
                    'window': tf.train.Feature(float_list=tf.train.FloatList(value=window.numpy().flatten())),
                    'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label.numpy()])),
                    'time_preread_index': tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[time_preread_index.numpy()])),
                    'window_index': tf.train.Feature(int64_list=tf.train.Int64List(value=[window_index.numpy()]))
                }
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())
        print(f"{gesture_number}号手势数据库中的元素格式:", dataset_train.element_spec)
    config.feature_shape = window_data_feature_train[1].shape

def online_data_contact():
    print("数据集整理开始")
    for turn in ["test", "train", "val"]:
        window_feature_data = []
        labels = []
        time_preread_indices = []
        window_indices = []
        for gesture_number in cf.gesture:
            data, label, time_preread_index, window_index = read_tfrecord(
                cf.folder_path + f"data/data_{gesture_number}_{turn}.tfrecord")
            window_feature_data = window_feature_data + data
            labels = labels + label
            time_preread_indices = time_preread_indices + time_preread_index
            window_indices = window_indices + window_index
        window_data_feature_tensor = tf.convert_to_tensor(window_feature_data, dtype=tf.float32)
        label_tensor = tf.convert_to_tensor(labels, dtype=tf.int32)
        time_preread_index_tensor = tf.convert_to_tensor(time_preread_indices, dtype=tf.int32)
        window_index_tensor = tf.convert_to_tensor(window_indices, dtype=tf.int32)
        dataset = tf.data.Dataset.from_tensor_slices(
            (window_data_feature_tensor, label_tensor, time_preread_index_tensor, window_index_tensor))
        # 确保保存路径存在
        save_path = cf.folder_path + "data"
        with tf.io.TFRecordWriter(os.path.join(save_path, f'data_contact_{turn}.tfrecord')) as writer:
            for window, label, time_preread_index, window_index in dataset:
                feature = {
                    'window': tf.train.Feature(float_list=tf.train.FloatList(value=window.numpy().flatten())),
                    'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label.numpy()])),
                    'time_preread_index': tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[time_preread_index.numpy()])),
                    'window_index': tf.train.Feature(int64_list=tf.train.Int64List(value=[window_index.numpy()]))
                }
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())
        # 打印当前数据集的个数
        print(f"{turn} 数据集的个数: {len(labels)}")
    print("数据处理完毕，开始训练模型")


