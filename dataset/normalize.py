#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2024/11/14 19:48
# @Author : 李 嘉 轩
# @team member : 赵雨新
# @File : script.py
# @Software: PyCharm Vscode
import numpy as np
def z_score_normalize(data):
    mean_vals = np.mean(data, axis=-2, keepdims=True)
    std_vals = np.std(data, axis=-2, keepdims=True)
    normalized_data = (data - mean_vals) / (std_vals+0.000001)
    return normalized_data