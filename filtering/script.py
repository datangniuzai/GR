#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2024/11/14 19:36
# @Author : 李 嘉 轩
# @File : script.py
# @Software: PyCharm


import numpy as np
import pandas as pd
from scipy import signal
import config as cf

def bandpass_and_notch_filter(data):
    """
    :param data: 要进行滤波的矩阵,行为通道数，列为采样点数
    :return: 滤波后的矩阵
    """
    channels = data.shape[0]  # 获取通道数
    N = data.shape[1]  # 获取数据列数
    # 对每一个通道的数据应用滤波器
    filtered_data = np.zeros((channels, N))
    # 设置带通滤波器
    sos = signal.butter(16, [25, 350], analog=False, btype='band', output='sos', fs=cf.sample_rate)
    for i in range(channels):
        filtered_data[i, :] = signal.sosfiltfilt(sos, data[i, :])
    # 应用陷波滤波器
    b, a = signal.iirnotch(50, 50, cf.sample_rate)
    for i in range(channels):
        filtered_data[i, :] = signal.filtfilt(b, a, filtered_data[i, :])
    b, a = signal.iirnotch(100, 50, cf.sample_rate)
    for i in range(channels):
        filtered_data[i, :] = signal.filtfilt(b, a, filtered_data[i, :])
    b, a = signal.iirnotch(150, 50, cf.sample_rate)
    for i in range(channels):
        filtered_data[i, :] = signal.filtfilt(b, a, filtered_data[i, :])
    b, a = signal.iirnotch(200, 50, cf.sample_rate)
    for i in range(channels):
        filtered_data[i, :] = signal.filtfilt(b, a, filtered_data[i, :])
    b, a = signal.iirnotch(250, 50, cf.sample_rate)
    for i in range(channels):
        filtered_data[i, :] = signal.filtfilt(b, a, filtered_data[i, :])
    return filtered_data

def filter_and_save_data():
    """
    将滤波后的数据进行保存
    """
    for gesture_number in cf.gesture:
        # 构造文件路径
        input_path = cf.folder_path + f'output_data/sEMG_data{gesture_number}.csv'
        output_path = cf.folder_path + f"process_data/filtered_data{gesture_number}.csv"
        # 读取数据
        df = pd.read_csv(input_path, header=None).to_numpy().T
        for i in range(cf.turn_read_sum):
            filter_pro_data = df[:,i * (cf.time_preread * cf.sample_rate):(i + 1) * (cf.time_preread * cf.sample_rate)]
            # 滤波
            filtered_data = bandpass_and_notch_filter(filter_pro_data)
            # 滤波数据
            # 保存滤波后的数据
            with open(output_path, 'a') as f:
                np.savetxt(f, filtered_data.T, delimiter=',', fmt='%.6f')
        print(f"{gesture_number}号手势数据已处理完毕")