#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2024/11/14 19:48
# @Author : Jason.LI
# @File : filtering.py
# @Software: PyCharm

import os
import time
from typing import List, Tuple
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import tensorflow as tf
from numpy.lib.stride_tricks import as_strided

from gesture_recognition.base.calculate_features import mav, mse, zc, wamp, rms
from gesture_recognition.base.filtering import bandpass_and_notch_filter
from base_config.config import GlobalConfig

# ------------------------------ #
#   Feature Extraction Function  #
# ------------------------------ #


def tfrecord_establish(
        df: np.ndarray,
        gesture_number: int,
        dataset_type: str,
        data_indices:list,
        once_read_time:int,
        sample_rate:int,
        window_size:int,
        step_size:int,
        window_size_little:int,
        step_size_little:int,
        path_to_use_data:str,
        max_workers:int = 10
):
    """
    General data processing function for feature extraction and saving for training, testing, and validation datasets.

    This function extracts features from input signals, processes them, and saves them as TensorFlow TFRecord files.
    """

    window_data_features = []
    window_data_labels = []

    tasks = [
        (
            rt,
            dataset_type,
            df,
            gesture_number,
            once_read_time,
            sample_rate,
            window_size,
            step_size,
            window_size_little,
            step_size_little,
        )
        for rt in data_indices
    ]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(process_read_time, tasks)

        for f, l in results:
            window_data_features.extend(f)
            window_data_labels.extend(l)

    window_data_feature_tensor = tf.convert_to_tensor(window_data_features, dtype=tf.float32)
    label_tensor = tf.convert_to_tensor(window_data_labels, dtype=tf.uint8)

    dataset = tf.data.Dataset.from_tensor_slices((window_data_feature_tensor, label_tensor))

    save_path = os.path.join(path_to_use_data, "processed_data")
    os.makedirs(save_path, exist_ok=True)

    tfrecord_path = os.path.join(save_path, f"data_{gesture_number}_{dataset_type}.tfrecord")
    tfrecord_save(dataset, tfrecord_path)

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
        window_data = bandpass_and_notch_filter( data = window_data)
        features.append(primary_window_feature(window_data, secondary_window_size, secondary_step_size))
        labels.append(gesture_number - 1)

    return features, labels


def primary_window_feature(data: np.ndarray, secondary_window_size: int, secondary_step_size: int) -> np.ndarray:
    """
    Efficiently split the input data into primary sliding windows.

    :param data: Input data matrix, shape (signal_length, num_channels)
    :param secondary_window_size: Size of each secondary_window
    :param secondary_step_size: Step size between secondary_window
    :return: Windows with extracted features, shape (num_windows, secondary_window_size, num_channels)
    """
    signal_length, num_channels = data.shape

    num_windows = (signal_length - secondary_window_size) // secondary_step_size + 1

    striped_shape = (num_windows, secondary_window_size, num_channels)
    striped_strides = (secondary_step_size * data.strides[0], data.strides[0], data.strides[1])
    windows = as_strided(data, shape=striped_shape, strides=striped_strides)

    features = z_score_normalize_per_timestep(np.apply_along_axis(secondary_window_feature, 1, windows))

    # features = np.transpose(features, (0, 2, 1))

    return features


def secondary_window_feature(data: np.ndarray) -> np.ndarray:
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

def write_td_data_info(
        path_to_use_data:str,
        train_indices:list,
        test_indices:list,
        val_indices:list,
        gesture_sequence:list,
        window_size:int,
        step_size:int,
        window_size_little:int,
        step_size_little:int
    ):
    """
    logging all parameters to tf_data_info.txt.
    """

    info_file_path = os.path.join(path_to_use_data, "processed_data", "tf_data_info.txt")

    with open(info_file_path, 'w') as f:
        f.write("Data Processing Parameters:\n")
        f.write(f"train_indices: {train_indices}\n")
        f.write(f"test_indices: {test_indices}\n")
        f.write(f"val_indices: {val_indices}\n")
        f.write(f"gesture_sequence: {gesture_sequence}\n")
        f.write(f"window_size: {window_size}\n")
        f.write(f"step_size: {step_size}\n")
        f.write(f"window_size_little: {window_size_little}\n")
        f.write(f"step_size_little: {step_size_little}\n")
        f.write("\nGenerated at: " + time.strftime("%Y-%m-%d %H:%M:%S") + "\n")

# ------------------------------ #
#   Tfrecord Build Function      #
# ------------------------------ #


def database_create(
        train_indices:list,
        test_indices:list,
        val_indices:list,
        gesture_sequence:list,
        path_to_use_data:str,
        once_read_time:int,
        sample_rate:int,
        window_size:int,
        step_size:int,
        window_size_little:int,
        step_size_little:int
        ):
    """
    Process the data and create the corresponding TFRecord files for training, testing, and validation datasets.

    Functionality:
    This function reads sEMG data from CSV files for each gesture, processes it, and saves it as TFRecord files for each dataset type (train, test, val).
    """
    print("Processing the data, please wait...")
    print(
        f"Using the {train_indices}-th data collection as the training set,\n"
        f"Using the {test_indices}-th data collection as the test set,\n"
        f"Using the {val_indices}-th data collection as the validation set.\n"
    )

    write_td_data_info(path_to_use_data,
        train_indices,
        test_indices,
        val_indices,
        gesture_sequence,
        window_size,
        step_size,
        window_size_little,
        step_size_little
    )

    for gesture_number in gesture_sequence:

        path = path_to_use_data + f"original_data/sEMG_data{gesture_number}.csv"
        df = pd.read_csv(path, header=None).to_numpy()
        for dataset_type, data_indices in [("train",train_indices), ("val",val_indices),( "test",test_indices)]:
            tfrecord_establish(df = df,
                               gesture_number=gesture_number,
                               dataset_type=dataset_type,
                               data_indices=data_indices,
                               once_read_time = once_read_time,
                               sample_rate = sample_rate,
                               window_size = window_size,
                               step_size =step_size,
                               window_size_little = window_size_little,
                               step_size_little = step_size_little,
                               path_to_use_data = path_to_use_data,
                                )
        print(f"Gesture {gesture_number} data processing completed.")


def tfrecord_connect( gesture_sequence:list, path_to_use_data:str):

    for dataset_type in ["train", "test", "val"]:

        merged_dataset = None

        for gesture_number in gesture_sequence:

            dataset = load_tfrecord_to_dataset(
                path_to_use_data + f"processed_data/data_{gesture_number}_{dataset_type}.tfrecord"
            )

            if merged_dataset is None:
                merged_dataset = dataset
            else:
                merged_dataset = merged_dataset.concatenate(dataset)

        connect_tfrecord_save_path = os.path.join(path_to_use_data, f"processed_data/data_contact_{dataset_type}.tfrecord")

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


def _parse_function(proto: tf.Tensor, feature_shape: list) -> Tuple[tf.Tensor, tf.Tensor]:
    """Parses TFRecord example into (features, label) tensors.

    Args:
        proto: Serialized TFRecord data
        feature_shape: Shape of the feature data

    Returns:
        Tuple of (window_data, label) tensors
    """
    keys_to_features = {
        "window": tf.io.FixedLenFeature(feature_shape, tf.float32),
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

    cf = GlobalConfig()
    cf.config_init()

