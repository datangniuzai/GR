#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/3/13 12:56
# @Author : Jason.LI
# @File : calculate_features.py
# @Software: PyCharm

import numpy as np
from scipy.stats import kurtosis
from statsmodels.tsa.api import VAR
from scipy.stats import skew as scipy_skew

# --------------------------------------------------
# Basic Statistical Features
# --------------------------------------------------


def mav(windowed_data: np.ndarray) -> np.ndarray:
    """
    Mean Absolute Value (MAV): Computes the mean of the absolute values of the signal.
    """
    return np.mean(np.abs(windowed_data), axis=0)


def sd(windowed_data: np.ndarray) -> np.ndarray:
    """
    Standard Deviation (SD): Computes the standard deviation of the signal.
    """
    return np.std(windowed_data, axis=0, ddof=1)


def var(windowed_data: np.ndarray) -> np.ndarray:
    """
    Variance (VAR): Computes the variance of the signal.
    """
    return np.var(windowed_data, axis=0, ddof=1)


def rms(windowed_data: np.ndarray) -> np.ndarray:
    """
    Root Mean Square (RMS): Computes the root mean square value of the signal.
    """
    return np.sqrt(np.mean(windowed_data**2, axis=0))


def mad(windowed_data: np.ndarray) -> np.ndarray:
    """
    Mean Absolute Deviation (MAD): Computes the mean absolute deviation of the signal.
    """
    return np.mean(np.abs(windowed_data - np.mean(windowed_data, axis=0)), axis=0)


def iqr(windowed_data: np.ndarray) -> np.ndarray:
    """
    Interquartile Range (IQR): Computes the interquartile range of the signal.
    """
    q75 = np.nanpercentile(windowed_data, 75, axis=0)
    q25 = np.nanpercentile(windowed_data, 25, axis=0)
    return q75 - q25


def kurt(windowed_data: np.ndarray) -> np.ndarray:
    """
    Kurtosis (KURT): Computes the kurtosis of the signal using Fisher's definition.
    """
    return kurtosis(windowed_data, axis=0, fisher=True, bias=False)


def skew(windowed_data: np.ndarray) -> np.ndarray:
    """
    Skewness (SKEW): Computes the skewness of the signal.
    """
    return scipy_skew(windowed_data, axis=0, bias=False)


def cov(windowed_data: np.ndarray) -> np.ndarray:
    """
    Coefficient of Variation (COV): Computes the coefficient of variation (std/mean).
    """
    return np.divide(np.std(windowed_data, axis=0), np.mean(windowed_data, axis=0))


# --------------------------------------------------
# Amplitude-Based Features
# --------------------------------------------------


def aac(windowed_data: np.ndarray) -> np.ndarray:
    """
    Average Amplitude Change (AAC): Computes the mean absolute difference of the signal.
    """
    return np.mean(np.abs(np.diff(windowed_data, axis=0)), axis=0)


def dasdv(windowed_data: np.ndarray) -> np.ndarray:
    """
    Difference Absolute Standard Deviation Value (DASDV): Computes the standard deviation of absolute differences.
    """
    return np.std(np.abs(np.diff(windowed_data, axis=0)), axis=0)


def dvarv(windowed_data: np.ndarray) -> np.ndarray:
    """
    Difference Variance Value (DVARV): Computes the variance of signal differences.
    """
    return np.var(np.diff(windowed_data, axis=0), axis=0)


def mmav(windowed_data: np.ndarray) -> np.ndarray:
    """
    Modified Mean Absolute Value (MMAV): Computes a weighted mean absolute value.
    """
    n = windowed_data.shape[0]
    weights = np.where((np.arange(n) >= 0.25 * n) & (np.arange(n) <= 0.75 * n), 1.0, 0.5)
    return np.mean(weights[:, np.newaxis] * np.abs(windowed_data), axis=0)


def mmav2(windowed_data: np.ndarray) -> np.ndarray:
    """
    Modified Mean Absolute Value 2 (MMAV2): Computes an advanced weighted mean absolute value.
    """
    n = windowed_data.shape[0]
    indices = np.arange(n)

    weights = np.ones(n)
    weights[indices < 0.25 * n] = (4 * indices[indices < 0.25 * n]) / n
    weights[indices > 0.75 * n] = 4 * (indices[indices > 0.75 * n] - n) / n

    return np.mean(weights[:, np.newaxis] * np.abs(windowed_data), axis=0)


def iemg(windowed_data: np.ndarray) -> np.ndarray:
    """
    Integrated EMG (IEMG): Computes the sum of absolute values of the signal.
    """
    return np.sum(np.abs(windowed_data), axis=0)


def wamp(windowed_data: np.ndarray, willison_threshold: float = 0.1) -> np.ndarray:
    """Calculate Willison Amplitude (WAMP) for each channel."""
    return np.sum(np.abs(np.diff(windowed_data, axis=0)) > willison_threshold, axis=0)


# --------------------------------------------------
# Zero-Based Features
# --------------------------------------------------


def zc(windowed_data: np.ndarray) -> np.ndarray:
    """Zero Crossing (ZC): Counts the number of times the signal crosses zero."""
    return np.sum(np.diff(np.sign(windowed_data), axis=0) != 0, axis=0)


def ssc(windowed_data: np.ndarray) -> np.ndarray:
    """Slope Sign Change (SSC): Counts changes in the slope sign."""
    return np.sum(np.diff(np.sign(np.diff(windowed_data, axis=0))) != 0, axis=0)


# --------------------------------------------------
# Energy-Based Features
# --------------------------------------------------


def ssi(windowed_data: np.ndarray) -> np.ndarray:
    """Simple Square Integral (SSI): Computes the sum of squared values of the signal."""
    return np.sum(windowed_data**2, axis=0)


def ae(windowed_data: np.ndarray) -> np.ndarray:
    """Average Energy (AE): Computes the mean of squared values of the signal."""
    return np.mean(windowed_data**2, axis=0)


def mse(windowed_data: np.ndarray) -> np.ndarray:
    """Calculate Mean Squared Error (MSE) for each channel."""
    return np.mean((windowed_data - np.mean(windowed_data, axis=0, keepdims=True)) ** 2, axis=0)


def vo(windowed_data: np.ndarray, order: int = 3) -> np.ndarray:
    """V-Order (VO): Computes the mean of the absolute signal raised to a given power."""
    return np.mean(np.abs(windowed_data) ** order, axis=0)


# --------------------------------------------------
# Complexity-Based Features
# --------------------------------------------------
def wl(windowed_data: np.ndarray) -> np.ndarray:
    """Waveform Length (WL): Computes the total waveform length of the signal."""
    return np.sum(np.abs(np.diff(windowed_data, axis=0)), axis=0)


def ltkeo(windowed_data: np.ndarray) -> np.ndarray:
    """Log Teager-Kaiser Energy Operator (LTKEO): Log-transformed Teager-Kaiser energy."""
    return np.log(np.mean(windowed_data[1:-1] ** 2 - windowed_data[:-2] * windowed_data[2:], axis=0) + 1e-9)


def mvar(windowed_data: np.ndarray, order: int = 0):
    """Compute multivariate AR (MVAR) model coefficients."""
    model = VAR(windowed_data)
    results = model.fit(order)
    return results.params


def ld(data: np.ndarray, epsilon: float = 1e-9) -> np.ndarray:
    """
    Log Detector (LD): Computes the log-energy-based feature.
    """
    return np.exp(np.mean(np.log(np.abs(data) + epsilon), axis=0))


def min_max_normalize(feature_vector: np.ndarray) -> np.ndarray:
    """Min-Max 归一化"""
    min_val = np.min(feature_vector)
    max_val = np.max(feature_vector)
    return (feature_vector - min_val) / (max_val - min_val + 1e-9)

if __name__ == '__main__':

    np.random.seed(42)
    data = np.random.randn(1000, 3)

    # 计算所有特征
    features = {
        "MAV": mav(data),
        "SD": sd(data),
        "VAR": var(data),
        "RMS": rms(data),
        "MAD": mad(data),
        "IQR": iqr(data),
        "KURT": kurt(data),
        "SKEW": skew(data),
        "COV": cov(data),
        "AAC": aac(data),
        "DASDV": dasdv(data),
        "DVARV": dvarv(data),
        "MMAV": mmav(data),
        "MMAV2": mmav2(data),
        "IEMG": iemg(data),
        "WAMP": wamp(data),
        "ZC": zc(data),
        "SSC": ssc(data),
        "SSI": ssi(data),
        "AE": ae(data),
        "MSE": mse(data),
        "VO (order=3)": vo(data, order=3),
        "WL": wl(data),
        "LTKEO": ltkeo(data),
        "LD": ld(data),
        "MVAR (order=0)": mvar(data, order=0)
    }

    # 归一化所有特征向量
    normalized_features = {}
    for feature_name, feature_vector in features.items():
        normalized_features[feature_name] = min_max_normalize(feature_vector)
        # normalized_features[feature_name] = z_score_normalize(feature_vector)


    # 打印归一化后的特征
    for feature_name, normalized_vector in normalized_features.items():
        print(f"{feature_name} (Normalized):", normalized_vector)

