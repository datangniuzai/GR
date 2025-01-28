#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2024/11/14 20:45
# @Author : Jiaxuan LI
# @File : train_process.py
# @Software: PyCharm

import os
import datetime
from typing import List

import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.callbacks import History
from sklearn.metrics import accuracy_score, confusion_matrix

import config as cf
from model_file import creat_model
from dataset import load_tfrecord_to_list,load_tfrecord_data_adjacency_label


class SaveModelPathCallback(tf.keras.callbacks.Callback):
    def __init__(self,model_save_path):
        super().__init__()
        self.model_save_path = model_save_path

    def on_epoch_end(self, epoch, logs=None):

        model_path = self.model_save_path.format(epoch=epoch + 1)

        cf.model_path = model_path

        print(f"Model saved at: {model_path}")

def make_train_folder() -> str:
    """
    Creates a training folder structure with subdirectories for saving pictures, models, error data,
    training information, test data, and figures. The folder is named with the current date and time.

    Returns:
    - str: The path of the main training folder created.
    """

    current_time = datetime.datetime.now()
    folder_name = current_time.strftime("%Y-%m-%d_%H-%M-%S")

    main_folder_path = os.path.join(cf.data_path, folder_name)
    os.makedirs(main_folder_path, exist_ok=True)

    picture_folder_path = os.path.join(main_folder_path, "picture")
    model_folder_path = os.path.join(main_folder_path, "models")
    error_data_information = os.path.join(main_folder_path, "error_data_information")
    training_information = os.path.join(main_folder_path, "training_information")
    test_folder_path = os.path.join(main_folder_path, "test")
    figures_folder_path = os.path.join(main_folder_path, "figures")

    os.makedirs(picture_folder_path, exist_ok=True)
    os.makedirs(model_folder_path, exist_ok=True)
    os.makedirs(error_data_information, exist_ok=True)
    os.makedirs(training_information, exist_ok=True)
    os.makedirs(test_folder_path, exist_ok=True)
    os.makedirs(figures_folder_path, exist_ok=True)

    return os.path.join(cf.data_path, folder_name) + "/"

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

def generate_unique_file_path(file_save_path: str, extension: str) -> str :

    existing_files = os.listdir(file_save_path)
    base_filename = "confusion_matrix"
    extension = extension
    counter = 1
    file_name = f"{base_filename}_{counter}{extension}"

    while file_name in existing_files:
        counter += 1
        file_name = f"{base_filename}_{counter}{extension}"

    new_file_path = os.path.join(file_save_path,file_name)

    return new_file_path

# --------------- #
#  Save Functions #
# --------------- #

def save_train_history(history: History) -> str:
    """
    Save the training history to a CSV file.

    :param history: History, History object generated during model training.
    :return: str, Path to the saved CSV file.
    """
    history_df = pd.DataFrame(history.history)

    training_info_csv_path = cf.training_info_path + "training_information/training_history.csv"

    history_df.to_csv(training_info_csv_path, index=False)

    return training_info_csv_path

def save_train_config() -> None:
    """
    Save training configuration details to a text file in cf.training_info_path.
    The file will be named "training_info.txt".
    """

    training_time = (cf.end_time - cf.start_time) / 60

    path_save_training_config = os.path.join(cf.training_info_path, f'training_information/training_info.txt')
    os.makedirs(os.path.dirname(path_save_training_config), exist_ok=True)

    with open(path_save_training_config, 'w') as file:
        file.write(f'Gesture numbers: {cf.gesture}\n')
        file.write(f'Dataset mode: {cf.tvt_select_mode}\n')
        file.write(f'Training time: {training_time:.2f} minutes\n')
        file.write(f'Training samples: {cf.train_num}\n')
        file.write(f'Test data locations: {cf.test_nums}\n')
        file.write(f'Validation data locations: {cf.val_nums}\n')
        file.write(f'Training data locations: {cf.train_nums}\n')
        file.write(f'Test samples: {cf.test_num}\n')
        file.write(f'Validation samples: {cf.val_num}\n')
        file.write(f'Window size: {cf.window_size}\n')
        file.write(f'Window step size: {cf.step_size}\n')
        file.write(f'Small window size: {cf.window_size_little}\n')
        file.write(f'Small window step size: {cf.step_size_little}\n')
        file.write(f'Epochs: {cf.epochs}\n')
        file.write(f'Scaling factor: {cf.scaling}\n')
        file.write(f'Model: {cf.model_name}\n')

    print(f"Total training time: {training_time:.2f} minutes")
    print("Training completed!")
    print(f"Saved training info to: {path_save_training_config}\n")

def save_test_info():
    # todo[1]: 将测试集在所有模型上的正确率补充进csv文件
    # todo[2]: 将recall率添加入csv文件
    pass

# --------------- #
#  Plot Functions #
# --------------- #
def plot_loss_acc(training_info_csv_path: str = None, fig_save_path: str = None) -> None:
    """
    This function plots training and validation loss and accuracy curves from a CSV file and saves the figure.

    Parameters:
    - csv_file_path (str, optional): Path to the CSV file containing training history data.
    - fig_save_path (str, optional): Path where the figure will be saved.
    """

    # Set default CSV file path if not provided
    if training_info_csv_path is None:
        if not hasattr(cf, 'training_info_path') or cf.training_info_path is None:
            raise ValueError("The 'training_info_path' is not set.")
        training_info_csv_path = os.path.join(cf.training_info_path, "training_information", "training_history.csv")

    # Set default figure save path if not provided
    if fig_save_path is None:
        if not hasattr(cf, 'training_info_path') or cf.training_info_path is None:
            raise ValueError("The 'training_info_path' is not set.")
        fig_save_path = os.path.join(cf.training_info_path, "figures", "training_history.svg")

    # Check if the CSV file exists
    if not os.path.exists(training_info_csv_path):
        raise FileNotFoundError(f"The CSV file '{training_info_csv_path}' was not found.")

    data = pd.read_csv(training_info_csv_path)

    loss = data['loss']
    accuracy = data['accuracy']
    val_loss = data['val_loss']
    val_accuracy = data['val_accuracy']

    plt.figure(figsize=(12, 4))

    # Plot training and validation accuracy
    plt.subplot(1, 2, 1)
    plt.plot(accuracy, label='Train Accuracy')
    plt.plot(val_accuracy, label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='upper left')

    # Plot training and validation loss
    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Train Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper left')

    os.makedirs(os.path.dirname(fig_save_path), exist_ok=True)
    plt.savefig(fig_save_path, format='svg')
    plt.close()

def plot_confusion_matrix(data_test_path: str = None, model_path: str = None, fig_save_path: str = None) -> None:
    if data_test_path is None:
        if hasattr(cf, 'data_path') and cf.data_path is not None:
            data_test_path = os.path.join(cf.data_path, "processed_data", "data_contact_test.tfrecord")
        else:
            raise ValueError("The 'data_path' is not set.")

    if model_path is None:
        if hasattr(cf, 'model_path') and cf.model_path is not None:
            model_path = cf.model_path
        else:
            raise ValueError("The 'model_path' is not set in either the argument or the configuration.")

    if fig_save_path is None:
        if hasattr(cf, 'training_info_path') and cf.training_info_path is not None:
            fig_save_path = os.path.join(cf.training_info_path, "figures")
        else:
            raise ValueError("The 'fig_save_path' is not set.")

    fig_save_path = generate_unique_file_path(file_save_path=fig_save_path, extension= ".svg")
    print("fig_save_path:",fig_save_path)
    cf.model.load_weights(model_path)

    tensor_x_test, tensor_adjacency_test, tensor_y_test = load_tfrecord_data_adjacency_label(data_test_path)

    y_pred_prob = cf.model.predict([tensor_adjacency_test, tensor_x_test])
    y_pred = np.argmax(y_pred_prob, axis=1)

    accuracy = accuracy_score(tensor_y_test, y_pred)
    cm = confusion_matrix(tensor_y_test, y_pred)
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
    plt.xlabel('Predicted label', fontsize=14)
    plt.ylabel('True label', fontsize=14)
    plt.title(f'Accuracy: {accuracy * 100:.2f}%', fontsize=16)

    plt.savefig(fig_save_path, format='svg')
    plt.close()

# ---------------- #
#  Train Functions #
# ---------------- #

def model_train():

    cf.training_info_path = make_train_folder()

    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    x_val, adjacency_val, y_val, *unused = load_tfrecord_to_list(cf.data_path + "processed_data/data_contact_val.tfrecord")
    x_train, adjacency_train, y_train, *unused = load_tfrecord_to_list(cf.data_path + "processed_data/data_contact_train.tfrecord")
    train_dataset = tf.data.Dataset.from_tensor_slices(((adjacency_train,x_train),y_train)).shuffle(len(x_train)).batch(32)
    val_dataset = tf.data.Dataset.from_tensor_slices(((adjacency_val,x_val),y_val)).batch(16)

    model_save_path = cf.training_info_path + f'models/model_' + '{epoch:02d}.keras'
    save_model_path_callback=SaveModelPathCallback(model_save_path)
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=model_save_path,
        save_weights_only=False,
        save_best_only=False,
        verbose=1
    )

    history = cf.model.fit(train_dataset, validation_data=val_dataset, epochs=cf.epochs,
                        callbacks=[model_checkpoint,save_model_path_callback])

    cf.training_info_csv_path = save_train_history(history)
    models_folder_path = os.path.join(cf.training_info_path, "models")
    fig_save_path = os.path.join(cf.training_info_path, "figures")
    test_all_models(models_folder_path, fig_save_path)

def test_all_models(models_folder_path: str,fig_save_path: str) -> None:

    if cf.model is None:
        cf.model = creat_model()

    models_list = get_models_list(models_folder_path)

    for model in models_list:
        model_path = os.path.join(models_folder_path, model)
        plot_confusion_matrix(model_path=model_path,fig_save_path=fig_save_path)