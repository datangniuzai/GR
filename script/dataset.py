#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2024/11/14 19:48
# @Author : Jason.LI
# @File : filtering.py
# @Software: PyCharm

import os
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import tensorflow as tf
from numpy.lib.stride_tricks import as_strided
from concurrent.futures import ProcessPoolExecutor

import config as cf
from filtering import bandpass_and_notch_filter
from calculate_features import mav, mse, zc, wamp, rms

# ------------------------------ #
#   Feature Extraction Function  #
# ------------------------------ #


def tfrecord_establish(df: np.ndarray, gesture_number: int, dataset_type: str):
    """
    General data processing function for feature extraction and saving for training, testing, and validation datasets.

    This function extracts features from input signals, processes them, and saves them as TensorFlow TFRecord files.

    :param df: Input signal data (shape: [num_channels, signal_length])
    :param gesture_number: Gesture identifier (integer)
    :param dataset_type: Type of dataset ('train'/'test'/'val')
    """

    if dataset_type == "train":
        read_times_list = cf.train_nums
    elif dataset_type == "test":
        read_times_list = cf.test_nums
    elif dataset_type == "val":
        read_times_list = cf.val_nums
    else:
        raise ValueError(f"Invalid dataset_type: {dataset_type}. Expected 'train', 'test', or 'val'.")

    window_data_features = []
    window_data_labels = []

    tasks = [
        (
            rt,
            dataset_type,
            df,
            gesture_number,
            cf.time_preread,
            cf.sample_rate,
            cf.window_size,
            cf.step_size,
            cf.window_size_little,
            cf.step_size_little,
        )
        for rt in read_times_list
    ]

    with ProcessPoolExecutor(max_workers=10) as executor:
        results = executor.map(process_read_time, tasks)

        for f, l in results:
            window_data_features.extend(f)
            window_data_labels.extend(l)

    window_data_feature_tensor = tf.convert_to_tensor(window_data_features, dtype=tf.float32)
    label_tensor = tf.convert_to_tensor(window_data_labels, dtype=tf.uint8)

    dataset = tf.data.Dataset.from_tensor_slices((window_data_feature_tensor, label_tensor))

    save_path = os.path.join(cf.data_path, "processed_data")
    os.makedirs(save_path, exist_ok=True)

    tfrecord_path = os.path.join(save_path, f"data_{gesture_number}_{dataset_type}.tfrecord")
    tfrecord_save(dataset, tfrecord_path)

    cf.feature_shape = window_data_features[0].shape


def process_read_time(args):
    """Wrapper function for parallel processing of a single read_time"""
    (
        read_time,
        dataset_type,
        df,
        gesture_number,
        time_preread,
        sample_rate,
        window_size,
        step_size,
        secondary_window_size,
        secondary_step_size,
    ) = args

    features = []
    labels = []
    start = (read_time - 1) * (time_preread * sample_rate)
    end = read_time * (time_preread * sample_rate)
    single_acquire_data = df[start:end, :]

    for j in range(0, single_acquire_data.shape[0] - window_size + 1, step_size):
        window_data = single_acquire_data[j : j + window_size, :]
        window_data = bandpass_and_notch_filter(window_data)
        features.append(primary_windows(window_data, secondary_window_size, secondary_step_size))
        labels.append(gesture_number - 1)

    return features, labels


def primary_windows(data: np.ndarray, secondary_window_size: int, secondary_step_size: int) -> np.ndarray:
    """
    Efficiently split the input data into primary sliding windows.

    :param data: Input data matrix, shape (signal_length, num_channels)
    :param secondary_window_size: Size of each secondary_window
    :param secondary_step_size: Step size between secondary_window
    :return: Windows with extracted features, shape (num_windows, secondary_window_size, num_channels)
    """
    signal_length, num_channels = data.shape

    num_windows = (signal_length - secondary_window_size) // secondary_step_size + 1

    strided_shape = (num_windows, secondary_window_size, num_channels)
    strided_strides = (secondary_step_size * data.strides[0], data.strides[0], data.strides[1])
    windows = as_strided(data, shape=strided_shape, strides=strided_strides)

    primary_window_feature = z_score_normalize_per_timestep(np.apply_along_axis(secondary_features, 1, windows))

    return primary_window_feature


def secondary_features(data: np.ndarray) -> np.ndarray:
    """
    extract the features from primary windowed_data.

    :param data: data of primary windowed_data，shape: (num_windows, window_size, num_channels)
    :return: the matrix of data's features，shape: (num_windows, num_channels, num_features)
    """
    features = np.array(
        [
            mav(data),
            rms(data),
            mse(data),
            zc(data),
            wamp(data),
        ]
    )

    return np.array(features)


def z_score_normalize_per_feature(features: np.ndarray) -> np.ndarray:
    normalized_features = (features - np.mean(features, axis=(0, 2), keepdims=True)) / np.std(
        features, axis=(0, 2), keepdims=True
    )
    return normalized_features


def z_score_normalize_per_channel(features: np.ndarray) -> np.ndarray:
    normalized_features = (features - np.mean(features, axis=(0, 1), keepdims=True)) / np.std(
        features, axis=(0, 1), keepdims=True
    )
    return normalized_features


def z_score_normalize_per_timestep(features: np.ndarray) -> np.ndarray:
    normalized_features = (features - np.mean(features, axis=(1, 2), keepdims=True)) / np.std(
        features, axis=(1, 2), keepdims=True
    )
    return normalized_features


def min_max_normalize_per_feature(features: np.ndarray) -> np.ndarray:
    features_min = np.min(features, axis=(0, 2), keepdims=True)
    features_max = np.max(features, axis=(0, 2), keepdims=True)
    return (features - features_min) / (features_max - features_min)


def min_max_normalize_per_channel(features: np.ndarray) -> np.ndarray:
    features_min = np.min(features, axis=(0, 1), keepdims=True)
    features_max = np.max(features, axis=(0, 1), keepdims=True)
    return (features - features_min) / (features_max - features_min)


def min_max_normalize_per_timestep(features: np.ndarray) -> np.ndarray:
    features_min = np.min(features, axis=(1, 2), keepdims=True)
    features_max = np.max(features, axis=(1, 2), keepdims=True)
    return (features - features_min) / (features_max - features_min)


# ------------------------------ #
#   Tfrecord Build Function      #
# ------------------------------ #


def database_create():
    """
    Process the data and create the corresponding TFRecord files for training, testing, and validation datasets.

    Functionality:
    This function reads sEMG data from CSV files for each gesture, processes it, and saves it as TFRecord files for each dataset type (train, test, val).
    """
    print("Processing the data, please wait...")
    print(
        f"Using the {cf.train_nums}-th data collection as the training set,\n"
        f"Using the {cf.test_nums}-th data collection as the test set,\n"
        f"Using the {cf.val_nums}-th data collection as the validation set.\n"
    )

    for gesture_number in cf.gesture:
        path = cf.data_path + f"original_data/sEMG_data{gesture_number}.csv"
        df = pd.read_csv(path, header=None).to_numpy()
        for dataset_type in ["train", "val", "test"]:
            tfrecord_establish(df, gesture_number, dataset_type)
        print(f"Gesture {gesture_number} data processing completed.")


def tfrecord_connect():

    for dataset_type in ["train", "test", "val"]:

        merged_dataset = None

        for gesture_number in cf.gesture:

            dataset = load_tfrecord_to_dataset(
                cf.data_path + f"processed_data/data_{gesture_number}_{dataset_type}.tfrecord"
            )

            if merged_dataset is None:
                merged_dataset = dataset
            else:
                merged_dataset = merged_dataset.concatenate(dataset)

        connect_tfrecord_save_path = os.path.join(cf.data_path, f"processed_data/data_contact_{dataset_type}.tfrecord")

        tfrecord_save(merged_dataset, connect_tfrecord_save_path)

        print(f"[{dataset_type}] data has been merged and saved at [{connect_tfrecord_save_path}]")

    print("data connection over")


def tfrecord_save(dataset: tf.data.Dataset, tfrecord_save_path: str):
    """
    Save the dataset as a TFRecord file.

    Parameters:
    dataset (iterable): A dataset containing window data, labels, and generated adjacency matrices.
    tfrecord_save_path (str): The path where the TFRecord file will be saved.

    Functionality:
    Converts each item in the dataset (window data, labels, etc.) to `tf.train.Example` format and writes it to the specified TFRecord file.
    """
    with tf.io.TFRecordWriter(tfrecord_save_path) as writer:
        for window, label in dataset:
            feature = {
                "window": tf.train.Feature(float_list=tf.train.FloatList(value=window.numpy().flatten())),
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label.numpy().item()])),
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())


# ----------------------------- #
#   Tfrecord Loading Function   #
# ----------------------------- #


def _parse_function(proto: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Parse each Example from the TFRecord file and adjust the data types and shapes.

    Parameters:
    proto (tf.Tensor): The input TFRecord data.

    Returns:
    Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]: Parsed data including window data, label.
    """
    keys_to_features: Dict[str, tf.io.FixedLenFeature] = {
        "window": tf.io.FixedLenFeature(cf.feature_shape, tf.float32),
        "label": tf.io.FixedLenFeature([1], tf.int64),
    }

    data = tf.io.parse_single_example(proto, keys_to_features)

    data["label"] = tf.cast(data["label"], tf.uint8)

    return data["window"], data["label"]


def load_tfrecord_to_dataset(tfrecord_path: str) -> tf.data.Dataset:
    """
    Load data from a TFRecord file.

    Parameters:
    tfrecord_path (str): Path to the TFRecord file.

    Returns:
    tf.data.Dataset: A TensorFlow dataset containing window data, labels.
    """

    dataset = tf.data.TFRecordDataset(tfrecord_path)

    dataset = dataset.map(_parse_function)

    return dataset


def load_tfrecord_to_list(tfrecord_path: str) -> Tuple[List[np.ndarray], List[int]]:
    """
    Load the TFRecord file and return the data as lists.

    Parameters:
    tfrecord_path (str): Path to the TFRecord file.

    Returns:
    Tuple[List[np.ndarray], List[np.ndarray], List[int]]:
        A tuple containing:
        - window_datas (List[np.ndarray]): List of window data arrays.
        - labels (List[int]): List of label integers.
    """
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(_parse_function)

    window_datas: List[np.ndarray] = []
    labels: List[int] = []

    for window_data, label in dataset:
        window_datas.append(window_data.numpy())
        labels.append(label.numpy())

    return window_datas, labels


def load_tfrecord_to_tensor(tfrecord_path: str) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Load the TFRecord file and return window data, adjacency matrix, labels,

    Parameters:
    tfrecord_path (str): Path to the TFRecord file.

    Returns:
    Tuple[tf.Tensor, tf.Tensor]: A tuple containing:
        - window_datas (tf.Tensor): Tensor containing window data.
        - labels (tf.Tensor): Tensor containing labels.
    """

    dataset = tf.data.TFRecordDataset(tfrecord_path)

    dataset = dataset.map(_parse_function)

    window_datas = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    labels = tf.TensorArray(dtype=tf.uint8, size=0, dynamic_size=True)

    for window_data, adjacency, label in dataset:
        window_datas = window_datas.write(window_datas.size(), window_data)
        labels = labels.write(labels.size(), label)

    window_datas = window_datas.stack()
    labels = labels.stack()

    return window_datas, labels


def load_tfrecord_data_label(tfrecord_path: str) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Load the TFRecord file and return window data, adjacency matrix, and labels as Tensors.

    Parameters:
    tfrecord_path (str): Path to the TFRecord file.

    Returns:
    Tuple[tf.Tensor, tf.Tensor, tf.Tensor]: A tuple containing:
        - window_datas (tf.Tensor): Tensor containing window data.
        - labels (tf.Tensor): Tensor containing labels.
    """
    dataset = tf.data.TFRecordDataset(tfrecord_path)

    dataset = dataset.map(_parse_function)

    window_datas = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    labels = tf.TensorArray(dtype=tf.uint8, size=0, dynamic_size=True)

    for window_data, label, *unused in dataset:
        window_datas = window_datas.write(window_datas.size(), window_data)
        labels = labels.write(labels.size(), label)

    window_datas = window_datas.stack()
    labels = labels.stack()

    return window_datas, labels


if __name__ == "__main__":
    cf.config_read()
