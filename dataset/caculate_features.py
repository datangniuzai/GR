#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2024/11/14 19:48
# @Author : 李 嘉 轩
# @team member : 赵雨新
# @File : script.py
# @Software: PyCharm Vscode
import numpy as np

def zero_crossing_rate(array):
    zero_crossings = np.sum(np.abs(np.diff(np.sign(array),axis=-1)) > 0,axis=-1)/10
    return zero_crossings
def mean_absolute_value(array):
    abs_array = np.abs(array)
    mean_abs = np.mean(abs_array,axis=-1)
    normalized_sum_abs_diff = mean_abs
    return normalized_sum_abs_diff/5
def root_mean_square(signal):
    rms = np.sqrt(np.mean(signal**2,axis=-1)) 
    normalized_sum_abs_diff = rms / 10
    return normalized_sum_abs_diff
def calculate_median_frequency(frequencies):
    median_freq = np.median(frequencies, axis=-1)
    return median_freq
def calculate_mean_frequency(frequencies, weights=None):
    if weights is None:
        weights = np.ones_like(frequencies)
    mean_freq = np.average(frequencies, weights=weights,axis=-1)
    return mean_freq 
def WL(data):
    sum_abs_diff = np.sum(np.abs(np.diff(data, axis=-1)), axis=-1)
    normalized_sum_abs_diff = sum_abs_diff / 1000
    return normalized_sum_abs_diff
