#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2024/11/14 19:36
# @Author : Jiaxuan LI
# @File : filtering.py
# @Software: PyCharm

import numpy as np
import pandas as pd
from scipy import signal

import config as cf

# Global definition of filter coefficients

# Bandpass filter (25Hz - 350Hz)
SOS_BANDPASS = signal.butter(16, [25, 350], analog=False, btype='band', output='sos', fs=cf.sample_rate)

# Notch filter coefficients (50Hz, 100Hz, 150Hz, 200Hz, 250Hz)
NOTCH_FILTERS = [signal.iirnotch(freq, 50, cf.sample_rate) for freq in [50, 100, 150, 200, 250]]

def bandpass_and_notch_filter(data: np.ndarray) -> np.ndarray:
    """
    Applies bandpass filtering and multiple notch filtering to the input data.

    :param data: Input matrix (shape: [channels, num_samples]) where rows represent channels
                 and columns represent sampled data points.
    :return: Filtered matrix (shape: [channels, num_samples]) after applying bandpass and notch filters.
    """
    channels, num_samples = data.shape

    filtered_data: np.ndarray = np.zeros((channels, num_samples))

    for i in range(channels):

        filtered_data[i, :] = signal.sosfiltfilt(SOS_BANDPASS, data[i, :])

        for b, a in NOTCH_FILTERS:
            filtered_data[i, :] = signal.filtfilt(b, a, filtered_data[i, :])

    return filtered_data

def filter_and_save_data() -> None:
    """
    Filters and saves the processed sEMG data for each gesture.

    :return: None
    """
    for gesture_number in cf.gesture:
        input_path: str = cf.data_path + f'output_data/sEMG_data{gesture_number}.csv'
        output_path: str = cf.data_path + f"process_data/filtered_data{gesture_number}.csv"

        df: np.ndarray = pd.read_csv(input_path, header=None).to_numpy().T

        for i in range(cf.turn_read_sum):
            start_idx: int = i * (cf.time_preread * cf.sample_rate)
            end_idx: int = (i + 1) * (cf.time_preread * cf.sample_rate)
            filter_pro_data: np.ndarray = df[:, start_idx:end_idx]

            filtered_data: np.ndarray = bandpass_and_notch_filter(filter_pro_data)

            with open(output_path, 'a') as f:
                np.savetxt(f, filtered_data.T, delimiter=',', fmt='%.6f')

        print(f"Data for gesture number {gesture_number} has been successfully processed")