#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2024/11/14 19:36
# @Author : Jason.LI
# @File : filtering.py
# @Software: PyCharm

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import signal

from base_config.config import GlobalConfig

# Global definition of filter coefficients
# Bandpass filter (20Hz - 500Hz)

SOS_BANDPASS = signal.butter(6, [20, 350], analog=False, btype='band', output='sos', fs= 2000)

sos_notch = []
for frequency in [50, 100, 150, 200, 250, 300]:
    b, a = signal.iirnotch(frequency, 50, 2000)
    sos = signal.tf2sos(b, a)
    sos_notch.append(sos)
SOS_NOTCH = np.concatenate(sos_notch, axis=0)

def bandpass_and_notch_filter(data: np.ndarray) -> np.ndarray:
    """
    Applies bandpass filtering and merged notch filtering with padding to reduce edge effects.

    :param data: Input matrix (shape: [num_samples, channels]).
    :return: Filtered matrix (shape: [num_samples, channels]).
    """
    if data.ndim != 2:
        raise ValueError(f"Expected 2D array, got {data.ndim}D")

    num_samples, channels = data.shape
    pad_length = num_samples
    padded_data = np.pad(data, ((pad_length, pad_length), (0, 0)), mode='reflect')
    filtered_data = np.zeros_like(padded_data)

    for i in range(channels):
        filtered_data[:, i] = signal.sosfiltfilt(SOS_BANDPASS, padded_data[:, i])
        filtered_data[:, i] = signal.sosfiltfilt(SOS_NOTCH, filtered_data[:, i])

    return filtered_data[pad_length:-pad_length, :]


class EMGFilter:
    """
    A professional-grade EMG signal processor implementing:
    - Bandpass filtering (20-350Hz default)
    - Multi-notch filtering (50Hz harmonics by default)
    - Edge artifact reduction via mirror padding

    """

    def __init__(self,
                 sample_rate: int,
                 bandpass_range: Tuple[float, float] = (20, 350),
                 notch_frequencies: Optional[List[float]] = None,
                 notch_q: float = 50.0):
        """
        Initialize EMG filter bank.

        Args:
            sample_rate: Sampling frequency in Hz
            bandpass_range: Cutoff frequencies (low, high) in Hz
            notch_frequencies: Frequencies to notch filter (default: 50Hz harmonics)
            notch_q: Quality factor for notch filters
        """
        self.sample_rate = sample_rate
        self.notch_q = notch_q
        self.bandpass_range = bandpass_range

        # Default to 50Hz harmonics if not specified
        self.notch_frequencies = notch_frequencies or [n * 50 for n in range(1, 7)]

        # Initialize filters
        self._sos_bandpass = self._design_bandpass()
        self._sos_notches = self._design_notches()

    def _design_bandpass(self):
        """Design 6th order Butterworth bandpass filter."""
        return signal.butter(
            N=6,
            Wn=self.bandpass_range,
            btype='band',
            analog=False,
            output='sos',
            fs=self.sample_rate
        )

    def _design_notches(self) -> np.ndarray:
        """Design cascaded IIR notch filters."""
        sos_list = []
        for freq in self.notch_frequencies:
            b_1, a_1 = signal.iirnotch(
                w0=freq,
                Q=self.notch_q,
                fs=self.sample_rate
            )
            sos_list.append(signal.tf2sos(b_1, a_1))
        return np.vstack(sos_list)

    def apply_filters(self,
                      data: np.ndarray,
                      padding_ratio: float = 1.0) -> np.ndarray:
        """
        Apply filtering pipeline to EMG data.

        Args:
            data: Input EMG [samples x channels]
            padding_ratio: Padding length relative to signal length (default 1.0)

        Returns:
            Filtered EMG data with same shape as input
        """
        # Input validation
        if data.ndim != 2:
            raise ValueError(f"Expected 2D array, got {data.ndim}D")

        num_samples, channels = data.shape

        pad_len = int(padding_ratio * num_samples)
        padded_data = np.pad(data, ((pad_len, pad_len), (0, 0)), mode='reflect')

        filtered = np.zeros_like(padded_data)

        for ch in range(channels):
            # Bandpass -> Notch cascade
            filtered[:, ch] = signal.sosfiltfilt(self._sos_bandpass, padded_data[:, ch])
            filtered[:, ch] = signal.sosfiltfilt(self._sos_notches, filtered[:, ch])

        # Remove padding
        return filtered[pad_len:-pad_len, :]

    def process_batch(self,
                      input_path: str,
                      output_path: str,
                      window_sec: float,
                      n_windows: int) -> None:
        """
        Process EMG data from CSV file in sliding windows.

        Args:
            input_path: Path to raw EMG CSV
            output_path: Destination path for filtered data
            window_sec: Analysis window length (seconds)
            n_windows: Number of windows to process
        """
        # Load data
        emg = pd.read_csv(input_path, header=None).values

        # Calculate samples per window
        win_samples = int(window_sec * self.sample_rate)

        # Process and save
        with open(output_path, 'w') as f:
            for i in range(n_windows):
                win = emg[i * win_samples: (i + 1) * win_samples, :]
                filtered = self.apply_filters(win)
                np.savetxt(f, filtered, delimiter=',', fmt='%.6f')

                # Separate windows with blank line
                if i < n_windows - 1:
                    f.write('\n')

    def filter_and_save_data(self,gesture_sequence,path_to_use_data,times_read_gesture,once_read_time) -> None:
        """
        Filters and saves the processed sEMG data for each gesture.

        :return: None
        """
        for gesture_number in gesture_sequence:
            input_path: str = path_to_use_data + f'original_data/sEMG_data{gesture_number}.csv'
            output_path: str = path_to_use_data + f"processed_data/filtered_data{gesture_number}.csv"

            df: np.ndarray = pd.read_csv(input_path, header=None).to_numpy()

            for i in range(times_read_gesture):
                start_idx: int = i * (once_read_time * self.sample_rate)
                end_idx: int = (i + 1) * (once_read_time * self.sample_rate)
                filter_pro_data: np.ndarray = df[start_idx:end_idx, :]

                filtered_data: np.ndarray = self.apply_filters(filter_pro_data)

                with open(output_path, 'a') as f:
                    np.savetxt(f, filtered_data, delimiter=',', fmt='%.6f')

            print(f"Data for gesture number {gesture_number} has been successfully processed")


if __name__ == '__main__':

    cf = GlobalConfig()
    cf.config_init()
    cf.display_config()

    filter_used = EMGFilter(sample_rate=cf.sample_rate)
    filter_used.filter_and_save_data(gesture_sequence=cf.gesture_sequence,path_to_use_data=cf.path_to_use_data,times_read_gesture=cf.times_read_gesture,once_read_time=cf.once_read_time)
