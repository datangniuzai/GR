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

# Global definition of filter coefficients

# Bandpass filter (25Hz - 350Hz)
SOS_BANDPASS = signal.butter(16, [25, 350], analog=False, btype='band', output='sos', fs=cf.sample_rate)

# Notch filter coefficients (50Hz, 100Hz, 150Hz, 200Hz, 250Hz)
NOTCH_FILTERS = [
    signal.iirnotch(freq, 50, cf.sample_rate) for freq in [50, 100, 150, 200, 250]
]

def bandpass_and_notch_filter(data:np.ndarray):
    """
    Applies bandpass filtering and multiple notch filtering to the input data.

    :param data: Input matrix where rows represent channels and columns represent sampled data points.
    :return: Filtered matrix after applying bandpass and notch filters.
    """
    channels, num_samples = data.shape

    # Initialize output data
    filtered_data = np.zeros((channels, num_samples))

    # Apply filtering to each channel
    for i in range(channels):

        filtered_data[i, :] = signal.sosfiltfilt(SOS_BANDPASS, data[i, :])

        for b, a in NOTCH_FILTERS:
            filtered_data[i, :] = signal.filtfilt(b, a, filtered_data[i, :])

    return filtered_data

def filter_and_save_data():
    """
    Save the filtered data.
    """
    for gesture_number in cf.gesture:

        input_path = cf.data_path + f'output_data/sEMG_data{gesture_number}.csv'
        output_path = cf.data_path + f"process_data/filtered_data{gesture_number}.csv"

        df = pd.read_csv(input_path, header=None).to_numpy().T

        for i in range(cf.turn_read_sum):

            filter_pro_data = df[:,i * (cf.time_preread * cf.sample_rate):(i + 1) * (cf.time_preread * cf.sample_rate)]

            filtered_data = bandpass_and_notch_filter(filter_pro_data)

            with open(output_path, 'a') as f:
                np.savetxt(f, filtered_data.T, delimiter=',', fmt='%.6f')

        print(f"{gesture_number}号手势数据已处理完毕")