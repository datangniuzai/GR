#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2024/11/14 20:45
# @Author : Jiaxuan LI
# @File : train_process.py
# @Software: PyCharm

import os
import csv
import datetime
import warnings
from typing import List

import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.callbacks import History
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score

import config as cf
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

def _call_all_models():
    # todo[1]提供一个方法反复加载模型
    pass
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
def save_test_confusion_matrix():
    # todo[1]: 将测试集在所有模型上的混淆矩阵进行计算保存
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

    if data_test_path is not None:
        if not hasattr(cf, 'data_path') or cf.data_path is None:
            raise ValueError("The 'data_path' is not set.")
        data_test_path = os.path.join(cf.data_path, "processed_data", "data_contact_test.tfrecord")

    if model_path is not None:
        if not hasattr(cf, 'model_path') or cf.model_path is None:
            raise ValueError("The 'model_path' is not set.")
        model_path = cf.model_path

    if fig_save_path is None:
        if not hasattr(cf, 'training_info_path') or cf.training_info_path is None:
            raise ValueError("The 'fig_save_path' is not set.")
        data_test_path = os.path.join(cf.training_info_path,"figures","confusion_matrix_test.svg")

    cf.model.load_weights(model_path)

    tensor_x_test, tensor_adjacency_test, tensor_y_test = load_tfrecord_data_adjacency_label(data_test_path)

    y_pred_prob = cf.model.predict([tensor_adjacency_test, tensor_x_test])
    y_pred = np.argmax(y_pred_prob, axis=1)

    accuracy = accuracy_score(tensor_y_test, y_pred)
    cm = confusion_matrix(tensor_y_test, y_pred)
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100

    # Do not display all zeros in the confusion matrix as annotations.
    annot = np.zeros_like(cm_perc, dtype=object)
    for i in range(cm_perc.shape[0]):
        for j in range(cm_perc.shape[1]):
            if cm_perc[i, j] != 0:
                annot[i, j] = f"{cm_perc[i, j]:.2f}"
            else:
                annot[i, j] = ""

    plt.figure(figsize=(15, 12))
    sns.heatmap(cm_perc, annot=annot, cmap="YlGnBu", fmt=".1f", linewidths=1, square=True,
                annot_kws={"fontsize": 12})
    plt.xlabel('Predicted label', fontsize=14)
    plt.ylabel('True label', fontsize=14)
    plt.title(f'Accuracy: {accuracy * 100:.2f}%', fontsize=16)
    plt.show()

    plt.savefig(fig_save_path, format='svg')
    plt.close()

def Plot_loos_acc_matrix_test(training_info_path= None):
    if not cf.training_info_path:
        warnings.warn("Warning: training_info_path is not set!")

    test_save_path = os.path.join(cf.training_info_path, "test")

    current_time = datetime.datetime.now()
    test_folder_name = current_time.strftime("%Y-%m-%d_%H-%M-%S")

    history_info_save_path = os.path.join(test_save_path, test_folder_name)
    os.makedirs(history_info_save_path, exist_ok=True)

    history_file_path = os.path.join(history_info_save_path, f'training_history.csv')

    x_test, adjacency_test,y_test, time_preread_indices, window_indices = load_tfrecord_to_list(
        cf.data_path + "processed_data/data_contact_test.tfrecord")
    x_test_tensor = tf.convert_to_tensor(x_test, dtype=tf.float32)
    adjacency_test_tensor = tf.convert_to_tensor(adjacency_test,dtype=tf.float32)
    print(adjacency_test_tensor.shape)
    # Retrieve and sort all files starting with "model_" in the directory;
    model_files = [f for f in os.listdir(cf.training_info_path + 'models/') if
                   f.startswith(f'model_')]

    # returns a sorted list of filenames.
    model_files.sort()

    picture_folder = os.path.join(history_info_save_path, 'picture')
    if not os.path.exists(picture_folder):
        os.makedirs(picture_folder)

    with open(history_file_path, 'w', newline='') as f:
        writer = csv.writer(f)

        header = ['epoch', 'loss', 'accuracy', 'val_loss', 'val_accuracy', 'test_accuracy']
        for i in range(cf.gesture_num):
            header.append(f'recall_gesture_{i}')
        writer.writerow(header)

        for epoch, model in enumerate(model_files):

            model_path = os.path.join(cf.training_info_path, 'models', model)

            cf.model.load_weights(model_path)

            y_pred_prob = cf.model.predict([adjacency_test_tensor,x_test_tensor])
            y_pred = np.argmax(y_pred_prob, axis=1)

            # calculate the accuracy
            accuracy = accuracy_score(y_test, y_pred)

            # calculate the recall
            recall = recall_score(y_test, y_pred, average=None)

            cm = confusion_matrix(y_test, y_pred)
            cm_sum = np.sum(cm, axis=1, keepdims=True)
            cm_perc = cm / cm_sum.astype(float) * 100

            # Do not display all zeros in the confusion matrix as annotations.
            annot = np.zeros_like(cm_perc, dtype=object)
            for i in range(cm_perc.shape[0]):
                for j in range(cm_perc.shape[1]):
                    if cm_perc[i, j] != 0:
                        annot[i, j] = f"{cm_perc[i, j]:.2f}"
                    else:
                        annot[i, j] = ""

            plt.figure(figsize=(15, 12))
            sns.heatmap(cm_perc, annot=annot, cmap="YlGnBu", fmt='', linewidths=1, square=True,
                        annot_kws={"fontsize": 12})
            plt.xlabel('Predicted label', fontsize=14)
            plt.ylabel('True label', fontsize=14)
            plt.title(f'Accuracy: {accuracy * 100:.2f}%', fontsize=16)

            heatmap_path = os.path.join(picture_folder, f'confusion_matrix_{model}.svg')

            plt.savefig(heatmap_path, format='svg')
            plt.close()

            # 记录所有预测错误的点所在的窗口位置
            failure_indices = np.where(y_pred != y_test)[0]
            failure_time_preread_indices = [time_preread_indices[i] for i in failure_indices]
            failure_window_indices = [window_indices[i] for i in failure_indices]
            failure_labels = [y_test[i] for i in failure_indices]
            failure_predicted_labels = [y_pred[i] for i in failure_indices]

            failure_file_path = os.path.join(history_info_save_path, f'failure_indices_{model}.csv')

            with open(failure_file_path, 'w', newline='') as failure_file:
                failure_writer = csv.writer(failure_file)
                failure_writer.writerow(['window_index', 'time_preread_index', 'label', 'predicted_label'])

                for window_index, time_preread_index, label, predicted_label in zip(failure_window_indices,
                                                                                    failure_time_preread_indices,
                                                                                    failure_labels,
                                                                                    failure_predicted_labels):
                    failure_writer.writerow([window_index, time_preread_index, label, predicted_label])

            row = [
                epoch + 1,
                cf.history.history['loss'][epoch],
                cf.history.history['accuracy'][epoch],
                cf.history.history['val_loss'][epoch],
                cf.history.history['val_accuracy'][epoch],
                accuracy
            ]
            row.extend(recall)
            writer.writerow(row)

def Plot_loos_acc_matrix():

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(cf.history.history['accuracy'])
    plt.plot(cf.history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.subplot(1, 2, 2)
    plt.plot(cf.history.history['loss'])
    plt.plot(cf.history.history['val_loss'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(cf.training_info_path + f'picture/loss_acc.svg', format='svg')
    plt.close()

    history_file_path = os.path.join(cf.training_info_path, f'training_information/training_history.csv')

    x_test, adjacency_test, y_test, read_indices_test, window_indices_test = load_tfrecord_to_list(
        cf.data_path + "processed_data/data_contact_test.tfrecord")
    x_test_tensor = tf.convert_to_tensor(x_test, dtype=tf.float32)
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

    model_files = [f for f in os.listdir(cf.training_info_path + 'models/') if
                   f.startswith(f'model_')]

    # returns a sorted list of filenames.
    model_files.sort()

    with open(history_file_path, 'w', newline='') as f:
        writer = csv.writer(f)

        header = ['epoch', 'loss', 'accuracy', 'val_loss', 'val_accuracy', 'test_accuracy']
        for i in range(cf.gesture_num):
            header.append(f'recall_gesture_{i}')
        writer.writerow(header)

        for epoch, model in enumerate(model_files):

            model_path = os.path.join(cf.training_info_path, 'models', model)

            cf.model.load_weights(model_path)

            y_pred_prob = cf.model.predict(x_test_tensor)
            y_pred = np.argmax(y_pred_prob, axis=1)

            # calculate the accuracy
            accuracy = accuracy_score(y_test, y_pred)

            # calculate the recall
            recall = recall_score(y_test, y_pred, average=None)

            cm = confusion_matrix(y_test, y_pred)
            cm_sum = np.sum(cm, axis=1, keepdims=True)
            cm_perc = cm / cm_sum.astype(float) * 100

            # Do not display all zeros in the confusion matrix as annotations.
            annot = np.zeros_like(cm_perc, dtype=object)
            for i in range(cm_perc.shape[0]):
                for j in range(cm_perc.shape[1]):
                    if cm_perc[i, j] != 0:
                        annot[i, j] = f"{cm_perc[i, j]:.2f}"
                    else:
                        annot[i, j] = ""

            plt.figure(figsize=(15, 12))
            sns.heatmap(cm_perc, annot=annot, cmap="YlGnBu", fmt='', linewidths=1, square=True,
                        annot_kws={"fontsize": 12})
            plt.xlabel('Predicted label', fontsize=14)
            plt.ylabel('True label', fontsize=14)
            plt.title(f'Accuracy: {accuracy * 100:.2f}%', fontsize=16)

            heatmap_path = os.path.join(cf.training_info_path, f'picture/confusion_matrix_{model}.svg')
            plt.savefig(heatmap_path, format='svg')
            plt.close()

            # 记录所有预测错误的点所在的窗口位置
            failure_indices = np.where(y_pred != y_test)[0]
            failure_time_preread_indices = [read_indices_test[i] for i in failure_indices]
            failure_window_indices = [window_indices_test[i] for i in failure_indices]
            failure_labels = [y_test[i] for i in failure_indices]
            failure_predicted_labels = [y_pred[i] for i in failure_indices]
            failure_file_path = os.path.join(cf.training_info_path,
                                             f'error_data_information/failure_indices_{model}.csv')

            with open(failure_file_path, 'w', newline='') as failure_file:
                failure_writer = csv.writer(failure_file)
                failure_writer.writerow(['window_index', 'time_preread_index', 'label', 'predicted_label'])

                for window_index, time_preread_index, label, predicted_label in zip(failure_window_indices,
                                                                                    failure_time_preread_indices,
                                                                                    failure_labels,
                                                                                    failure_predicted_labels):
                    failure_writer.writerow([window_index, time_preread_index, label, predicted_label])

            row = [
                epoch + 1,
                cf.history.history['loss'][epoch],
                cf.history.history['accuracy'][epoch],
                cf.history.history['val_loss'][epoch],
                cf.history.history['val_accuracy'][epoch],
                accuracy
            ]
            row.extend(recall)
            writer.writerow(row)

    cf.model_name = cf.model.name

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
