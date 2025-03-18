#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2024/11/14 19:36
# @Author : Jason.LI
# @File : filtering.py
# @Software: PyCharm

import os
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt

import config as cf

# Global definition of filter coefficients
# Bandpass filter (20Hz - 500Hz)

SOS_BANDPASS = signal.butter(6, [20, 350], analog=False, btype='band', output='sos', fs=cf.sample_rate)

sos_notch = []
for frequency in [50, 100, 150, 200, 250, 300]:
    b, a = signal.iirnotch(frequency, 50, cf.sample_rate)
    sos = signal.tf2sos(b, a)
    sos_notch.append(sos)
SOS_NOTCH = np.concatenate(sos_notch, axis=0)

def bandpass_and_notch_filter(data: np.ndarray) -> np.ndarray:
    """
    Applies bandpass filtering and merged notch filtering with padding to reduce edge effects.

    :param data: Input matrix (shape: [num_samples, channels]).
    :return: Filtered matrix (shape: [num_samples, channels]).
    """
    num_samples, channels = data.shape
    pad_length = num_samples
    padded_data = np.pad(data, ((pad_length, pad_length), (0, 0)), mode='reflect')
    filtered_data = np.zeros_like(padded_data)

    for i in range(channels):
        filtered_data[:, i] = signal.sosfiltfilt(SOS_BANDPASS, padded_data[:, i])
        filtered_data[:, i] = signal.sosfiltfilt(SOS_NOTCH, filtered_data[:, i])

    return filtered_data[pad_length:-pad_length, :]

def filter_and_save_data() -> None:
    """
    Filters and saves the processed sEMG data for each gesture.

    :return: None
    """
    for gesture_number in cf.gesture:
        input_path: str = cf.data_path + f'original_data/sEMG_data{gesture_number}.csv'
        output_path: str = cf.data_path + f"processed_data/filtered_data{gesture_number}.csv"

        df: np.ndarray = pd.read_csv(input_path, header=None).to_numpy()

        for i in range(cf.turn_read_sum):
            start_idx: int = i * (cf.time_preread * cf.sample_rate)
            end_idx: int = (i + 1) * (cf.time_preread * cf.sample_rate)
            filter_pro_data: np.ndarray = df[start_idx:end_idx, :]

            filtered_data: np.ndarray = bandpass_and_notch_filter(filter_pro_data)

            with open(output_path, 'a') as f:
                np.savetxt(f, filtered_data, delimiter=',', fmt='%.6f')

        print(f"Data for gesture number {gesture_number} has been successfully processed")

def plot_frequency_comparison(original_data: np.ndarray,
                              filtered_data: np.ndarray,
                              sample_rate: float,
                              channel_index: int = 0):

    original_signal = original_data[:, channel_index]
    filtered_signal = filtered_data[:, channel_index]

    n = len(original_signal)
    freq = np.fft.rfftfreq(n, d=1 / sample_rate)

    window = np.hanning(n)

    fft_original = np.fft.rfft(original_signal * window)
    psd_original = np.abs(fft_original) ** 2 / (sample_rate * np.sum(window ** 2))

    fft_filtered = np.fft.rfft(filtered_signal * window)
    psd_filtered = np.abs(fft_filtered) ** 2 / (sample_rate * np.sum(window ** 2))

    plt.figure(figsize=(12, 6))

    plt.semilogy(freq, psd_original, label='Original', alpha=0.7, color='blue')

    plt.semilogy(freq, psd_filtered, label='Filtered', alpha=0.7, color='red')

    plt.axvspan(20, 500, color='green', alpha=0.1, label='Bandpass Range')
    for notch_freq in [50, 100, 150, 200, 250, 300]:
        plt.axvline(notch_freq, color='grey', linestyle='--', alpha=0.5)

    plt.title(f'Frequency Spectrum Comparison (Channel {channel_index})')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density (dB/Hz)')
    plt.ylim(1e-12, 1e2)
    plt.xlim(0, sample_rate / 2)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.show()



