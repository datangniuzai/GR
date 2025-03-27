#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2024/11/14 20:45
# @Author : Jason.LI
# @File : train_process.py
# @Software: PyCharm

import csv
import datetime
import logging
import os
import re
import time
import warnings
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from keras.callbacks import History
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score
from tensorflow.keras import Model

from base_config.config import GlobalConfig
from gesture_recognition.base.dataset import load_tfrecord_to_list, load_tfrecord_data_label


class SaveModelPathCallback(tf.keras.callbacks.Callback):
    def __init__(self, model_save_path):
        super().__init__()
        self.model_save_path = model_save_path

    def on_epoch_end(self, epoch, logs=None):

        model_path = self.model_save_path.format(epoch=epoch + 1)

        print(f"Model saved at: {model_path}")

def make_train_folder(path_to_use_data: str, k: int = None, model_name: str = None) -> str:
    """
    Creates a training folder structure with subdirectories for saving pictures, models, test_information,
    training information, test data, and figures. The folder is named with the current date and time.

    Returns:
    - str: The path of the main training folder created.
    """

    folder_name = datetime.datetime.now().strftime("%m-%d_%H-%M")

    if k:
        folder_name = f"fold{k}_{folder_name}"

    if model_name:
        folder_name = f"{model_name}_{folder_name}"

    main_folder_path = os.path.join(path_to_use_data, "all_train_info", folder_name)
    os.makedirs(main_folder_path, exist_ok=True)

    model_folder_path = os.path.join(main_folder_path, "models")
    test_information = os.path.join(main_folder_path, "test_information")
    training_information = os.path.join(main_folder_path, "training_information")
    test_folder_path = os.path.join(main_folder_path, "test")
    figures_folder_path = os.path.join(main_folder_path, "figures")

    os.makedirs(model_folder_path, exist_ok=True)
    os.makedirs(test_information, exist_ok=True)
    os.makedirs(training_information, exist_ok=True)
    os.makedirs(test_folder_path, exist_ok=True)
    os.makedirs(figures_folder_path, exist_ok=True)

    return main_folder_path + "/"

def get_models_list(models_folder_path: str) -> List[str]:
    """
    Retrieve all file names from a specified folder.

    :param models_folder_path: str, Path to the folder.
    :return: List[str], List of file names in the folder.
    """
    try:
        file_names = [f for f in os.listdir(models_folder_path) if os.path.isfile(os.path.join(models_folder_path, f))]
        return file_names
    except FileNotFoundError:
        print(f"Error: Folder '{models_folder_path}' not found.")
        return []
    except PermissionError:
        print(f"Error: No permission to access folder '{models_folder_path}'.")
        return []

def generate_unique_file_path(base_filename: str, file_save_path: str, extension: str) -> str:

    existing_files = os.listdir(file_save_path)
    extension = extension
    counter = 1
    file_name = f"{base_filename}_{counter}{extension}"

    while file_name in existing_files:
        counter += 1
        file_name = f"{base_filename}_{counter}{extension}"

    new_file_path = os.path.join(file_save_path, file_name)

    return new_file_path

# --------------- #
#  Save Functions #
# --------------- #

def train_history_to_csv(train_history: History, train_folder_path: str) -> str:
    """
    Save the training history to a CSV file.

    :param train_history: History, History object generated during model training.
    :param train_folder_path: the training folder path.
    :return: str, Path to the saved CSV file.
    """
    history_df = pd.DataFrame(train_history.history)

    history_csv_file = train_folder_path + "training_information/training_history.csv"

    history_df.to_csv(history_csv_file, index=False)

    return history_csv_file

def train_config_to_txt(
    train_folder_path: str,
    start_train_time,
    end_train_time,
    train_duration_minutes,
    gesture: list,
    tvt_select_mode: str,
    train_num: int,
    val_num: int,
    test_num: int,
    test_indices: list,
    val_indices: list,
    train_indices: list,
    window_size: int,
    step_size: int,
    window_size_little: int,
    step_size_little: int,
    epochs: int,
    model_name: str,
) -> str:
    """
    Save training configuration details to a text file in train_folder_path/training_information.
    The file will be named "training_info.txt".
    """


    path_save_training_config = os.path.join(train_folder_path, f"training_information/training_config.txt")
    os.makedirs(os.path.dirname(path_save_training_config), exist_ok=True)

    with open(path_save_training_config, "w") as file:
        file.write(f"Gesture numbers: {gesture}\n")
        file.write(f"Dataset mode: {tvt_select_mode}\n")
        file.write(f"Model training start time:{start_train_time}\n")
        file.write(f"Model training end time:{end_train_time}\n")
        file.write(f"Training total time: {train_duration_minutes:.2f} minutes\n")
        file.write(f"Training samples: {train_num}\n")
        file.write(f"Validation samples: {val_num}\n")
        file.write(f"Test samples: {test_num}\n")
        file.write(f"Test data locations: {test_indices}\n")
        file.write(f"Validation data locations: {val_indices}\n")
        file.write(f"Training data locations: {train_indices}\n")
        file.write(f"Primary window size: {window_size}\n")
        file.write(f"Primary window step size: {step_size}\n")
        file.write(f"Secondary window size: {window_size_little}\n")
        file.write(f"Secondary window step size: {step_size_little}\n")
        file.write(f"Epochs: {epochs}\n")
        file.write(f"Model: {model_name}\n")

    print(f"Total training time: {train_duration_minutes:.2f} minutes")
    print("Training completed!")
    print(f"Saved training info to: {path_save_training_config}\n")

    return path_save_training_config

def test_result_to_csv(path_to_use_data:str,train_folder_path: str, gesture_sequence: list, feature_shape: list, model: Model = None) -> str:

    if model is None:
        raise ValueError("Model is not initialized. Please provide a valid model.")

    test_result_csv_file = os.path.join(train_folder_path, "test_information", "test_result.csv")
    path_test_data = os.path.join(path_to_use_data, "processed_data", "data_contact_test.tfrecord")
    models_folder_path = os.path.join(train_folder_path, "models")

    if os.path.exists(test_result_csv_file):
        while True:
            user_input = input(f"The file '{test_result_csv_file}' already exists. Do you want to delete it? (y/n): ")
            if user_input.lower() == "y":
                os.remove(test_result_csv_file)
                print(f"Deleted existing file: {test_result_csv_file}")
                break
            elif user_input.lower() == "n":
                # Add a suffix to the filename
                base_name, ext = os.path.splitext(test_result_csv_file)
                counter = 1
                while True:
                    new_name = f"{base_name}_{counter}{ext}"
                    if not os.path.exists(new_name):
                        test_result_csv_file = new_name
                        print(f"Using new filename: {test_result_csv_file}")
                        break
                    counter += 1
                break
            else:
                print("Invalid input. Please enter 'y' to delete or 'n' to cancel.")

    x_test, y_test = load_tfrecord_data_label(path_test_data, feature_shape)

    model_list = [f for f in os.listdir(models_folder_path) if f.endswith(".keras")]

    header = ["epoch", "accuracy"] + [f"recall_gesture_{gesture}" for gesture in gesture_sequence]

    with open(test_result_csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

    for one_model_name in model_list:
        match = re.search(r"_(\d+)\.keras", one_model_name)

        if match:
            model_number = match.group(1)
        else:
            raise ValueError(f"Model name '{one_model_name}' does not match the expected format '_<number>.keras'.")

        model_path = os.path.join(models_folder_path, one_model_name)

        model.load_weights(model_path)

        y_pred_prob = model.predict([x_test])

        y_pred = np.argmax(y_pred_prob, axis=1)

        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred, average=None)

        row = [model_number, accuracy] + list(recall)
        with open(test_result_csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)

    print(f"Test information saved to '{test_result_csv_file}'.")
    logging.info(f"Test information saved to '{test_result_csv_file}'.")

    return test_result_csv_file
# --------------- #
#  Plot Functions #
# --------------- #
def plot_loss_acc(train_folder_path: str, fig_save_path: str = None) -> None:
    """
    This function plots training and validation loss and accuracy curves from a CSV file and saves the figure.

    Parameters:
    - csv_file_path (str, optional): Path to the CSV file containing training history data.
    - fig_save_path (str, optional): Path where the figure will be saved.
    """

    training_history_csv_path = os.path.join(train_folder_path, "training_information", "training_history.csv")

    if fig_save_path is None:
        fig_save_path = os.path.join(train_folder_path, "figures", "training_history.svg")

    if not os.path.exists(training_history_csv_path):
        raise FileNotFoundError(f"The CSV file '{training_history_csv_path}' was not found.")

    data = pd.read_csv(training_history_csv_path)

    loss = data["loss"]
    acc = data["accuracy"]
    val_loss = data["val_loss"]
    val_acc = data["val_accuracy"]

    plt.figure(figsize=(12, 4))

    # Plot training and validation accuracy
    plt.subplot(1, 2, 1)
    plt.plot(acc, label="Train Accuracy")
    plt.plot(val_acc, label="Validation Accuracy")
    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc="upper left")

    # Plot training and validation loss
    plt.subplot(1, 2, 2)
    plt.plot(loss, label="Train Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="upper left")

    os.makedirs(os.path.dirname(fig_save_path), exist_ok=True)
    plt.savefig(fig_save_path, format="svg")
    plt.close()

    logging.info(f"The loss and acc plot has benn saved as {fig_save_path}")
    print(f"The loss and acc plot has benn saved as {fig_save_path}")

def plot_one_model_confusion_matrix(
    model: Model = None,
    path_test_data: str = None,
    model_path: str = None,
    fig_save_path: str = None,
    feature_shape: list = None,
) -> None:

    if model is None:
        raise ValueError("Model is not initialized. Please provide a valid model.")
    if fig_save_path is None:
        raise ValueError("The 'fig_save_path' is not set.")
    if path_test_data is None:
        raise ValueError("The 'path_test_data' is not set.")
    if model_path is None:
        raise ValueError("The 'model_path' is not set in either the argument or the configuration.")
    if feature_shape is None:
        raise ValueError("The 'feature_shape' is not set.")

    fig_name = generate_unique_file_path(
        base_filename="confusion_matrix", file_save_path=fig_save_path, extension=".svg"
    )

    model.load_weights(model_path)
    x_test, y_test = load_tfrecord_data_label(path_test_data, feature_shape)

    y_pred_prob = model.predict([x_test])
    y_pred = np.argmax(y_pred_prob, axis=1)

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100

    annot = np.zeros_like(cm_perc, dtype=object)
    for i in range(cm_perc.shape[0]):
        for j in range(cm_perc.shape[1]):
            if cm_perc[i, j] != 0:
                annot[i, j] = f"{cm_perc[i, j]:.2f}"
            else:
                annot[i, j] = ""

    plt.figure(figsize=(15, 12))
    sns.heatmap(cm_perc, annot=annot, cmap="YlGnBu", fmt="", linewidths=1, square=True, annot_kws={"fontsize": 12})
    plt.xlabel("Predicted label", fontsize=14)
    plt.ylabel("True label", fontsize=14)
    plt.title(f"Accuracy: {accuracy * 100:.2f}%", fontsize=16)
    plt.savefig(fig_name, format="svg")
    plt.close()

    print(f"fig is saved at{fig_save_path},and named as {fig_name}")
    logging.info(f"fig is saved at{fig_save_path},and named as {fig_name}")

def plot_all_models_confusion_matrix(
    model: Model, train_folder_path: str, path_to_use_data: str, feature_shape: list
) -> None:

    models_folder_path = os.path.join(train_folder_path, "models")
    fig_save_path = os.path.join(train_folder_path, "figures")
    path_test_data = os.path.join(path_to_use_data, "processed_data", "data_contact_val.tfrecord")

    models_list = get_models_list(models_folder_path)

    for one_model in models_list:
        model_path = os.path.join(models_folder_path, one_model)
        plot_one_model_confusion_matrix(
            model=model,
            path_test_data=path_test_data,
            model_path=model_path,
            fig_save_path=fig_save_path,
            feature_shape=feature_shape,
        )


# ---------------- #
#  Train Functions #
# ---------------- #


def one_model_train(
    model: Model,
    config: GlobalConfig,
    model_name: str,
    train_folder_path: str = None,

):

    if model_name is None:
        raise ValueError("The 'model_name' is not set.")

    config.update_param("model_name", model_name)

    if train_folder_path is None:
        warnings.warn("The 'train_folder_path' is not set.", UserWarning)
        train_folder_path = make_train_folder(path_to_use_data=config.path_to_use_data, model_name=model_name)
        config.update_param("train_folder_path", train_folder_path)
        logging.info(f"train_folder_path was set to {train_folder_path}")

    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices("GPU")))

    x_val, y_val, *unused = load_tfrecord_to_list(
        config.path_to_use_data + "processed_data/data_contact_val.tfrecord", config.feature_shape
    )
    x_train, y_train, *unused = load_tfrecord_to_list(
        config.path_to_use_data + "processed_data/data_contact_train.tfrecord", config.feature_shape
    )

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(len(x_train)).batch(32)
    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(16)

    model_save_path = config.train_folder_path + f"models/model_" + "{epoch:02d}.keras"

    save_model_path_callback = SaveModelPathCallback(model_save_path)

    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=model_save_path, save_weights_only=False, save_best_only=False, verbose=1
    )

    start_train_time = time.time()
    config.update_param("start_train_time",datetime.datetime.now())

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=config.epochs,
        callbacks=[model_checkpoint, save_model_path_callback],
    )

    end_train_time = time.time()
    config.update_param("end_train_time",datetime.datetime.now())

    history_csv_file = train_history_to_csv(history, train_folder_path)
    config.update_param("history_csv_file", history_csv_file)

    plot_loss_acc(config.train_folder_path)

    train_duration_seconds = end_train_time - start_train_time
    train_duration_minutes = train_duration_seconds / 60
    config.update_param("train_duration_minutes",train_duration_minutes)

    path_save_training_config = train_config_to_txt(
        config.train_folder_path,
        config.start_train_time,
        config.end_train_time,
        config.train_duration_minutes,
        config.gesture_sequence,
        config.tvt_select_mode,
        config.train_num,
        config.val_num,
        config.test_num,
        config.test_indices,
        config.val_indices,
        config.train_indices,
        config.window_size,
        config.step_size,
        config.window_size_little,
        config.step_size_little,
        config.epochs,
        config.model_name,
    )

    config.update_param("train_config_txt_path", path_save_training_config)
    plot_all_models_confusion_matrix(model, train_folder_path, config.path_to_use_data, config.feature_shape)

    train_config_txt_path = test_result_to_csv(
        train_folder_path= config.train_folder_path,
        gesture_sequence=config.gesture_sequence,
        feature_shape= config.feature_shape,
        model=model,
        path_to_use_data= config.path_to_use_data
    )
    config.update_param("train_config_txt_path", train_config_txt_path)


# def k_fold_cross_validation(k):
#     model_list = ["litestfnet", "cnn", "bilstm", "cnn-bilstm"]
#     model_list = ["litestfnet", "cnn", "bilstm", "cnn-bilstm"]
#
#     model_function_map = {
#         "bilstm": bilstm_model_creat,
#         "cnn-bilstm": cnn_bilstm_model_creat,
#         "litestfnet": litestfnet_model_creat,
#         "cnn": cnn_mode_creat,
#     }
#     for k_step in range(1, k + 1):
#          train_nums,  test_nums,  val_nums, _ = split_data(
#              turn_read_sum,  train_num,  test_num,  val_num
#         )
#         database_create()
#         tfrecord_connect()
#         for model_name in model_list:
#              training_info_path = make_train_folder(k=k_step, model_name=model_name)
#             model_function = model_function_map.get(model_name)
#             if model_function:
#                  model = model_function()
#                  training_info_path = make_train_folder(k_step, model_name)
#                 one_model_train()
#             else:
#                 print(f"Function for {model_name} not found.")
