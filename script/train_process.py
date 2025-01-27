#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2024/11/14 20:45
# @Author : Jiaxuan LI
# @File : train_process.py
# @Software: PyCharm

import csv
import datetime
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score

import config as cf
from dataset import load_tfrecord_to_tensor,load_tfrecord_to_list


def model_train():

    current_time = datetime.datetime.now()
    folder_name = current_time.strftime("%Y-%m-%d_%H-%M-%S")

    main_folder_path = os.path.join(cf.data_path, folder_name)
    os.makedirs(main_folder_path, exist_ok=True)

    picture_folder_path = os.path.join(main_folder_path, "Picture")
    model_folder_path = os.path.join(main_folder_path, "save_model")
    error_data_information = os.path.join(main_folder_path, "Error_Data_Information")
    training_information = os.path.join(main_folder_path, "Training_Information")
    test_folder_path = os.path.join(main_folder_path, "Test")

    os.makedirs(picture_folder_path, exist_ok=True)
    os.makedirs(model_folder_path, exist_ok=True)
    os.makedirs(error_data_information, exist_ok=True)
    os.makedirs(training_information, exist_ok=True)
    os.makedirs(test_folder_path, exist_ok=True)

    cf.training_info_path = cf.data_path + folder_name + "/"

    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        cf.training_info_path + f'save_model/model_' + '{epoch:02d}.keras',
        save_weights_only=False, save_best_only=False, verbose=1)

    x_val, adjacency_val,y_val, x_p , x_t = load_tfrecord_to_list(cf.data_path + "processed_data/data_contact_val.tfrecord")
    x_train, adjacency_train,y_train, v_p , v_t = load_tfrecord_to_list(cf.data_path + "processed_data/data_contact_train.tfrecord")
    train_dataset = tf.data.Dataset.from_tensor_slices(((adjacency_train,x_train),y_train)).shuffle(len(x_train)).batch(32)
    val_dataset = tf.data.Dataset.from_tensor_slices(((adjacency_val,x_val),y_val)).batch(16)
    cf.history = cf.model.fit(train_dataset, validation_data=val_dataset, epochs=cf.epochs,
                        callbacks=[model_checkpoint])

def plot_confusion_matrix(training_info_path= None,model_path= None):

    if training_info_path is not None:
        cf.training_info_path = training_info_path + "/"
    if model_path is not None:
        model_path = model_path

    cf.model.load_weights(model_path)

    data_test_path= cf.data_path + "processed_data/data_contact_test.tfrecord"

    tensor_x_test, tensor_adjacency_test, tensor_y_test, *unused = load_tfrecord_to_tensor(data_test_path)

    y_pred_prob = cf.model.predict([tensor_adjacency_test,tensor_x_test])
    y_pred = np.argmax(y_pred_prob, axis=1)

    accuracy = accuracy_score(tensor_y_test, y_pred)
    cm = confusion_matrix(tensor_y_test, y_pred)
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_perc, annot=True, cmap="YlGnBu", fmt=".1f", linewidths=.5, square=True,annot_kws={"fontsize": 12})
    plt.xlabel('Predicted label', fontsize=14)
    plt.ylabel('True label', fontsize=14)
    plt.title(f'Confusion Matrix (Accuracy: {accuracy * 100:.2f}%)', fontsize=16)
    plt.show()

def Plot_loos_acc_matrix_test():
    if not cf.training_info_path:
        warnings.warn("Warning: training_info_path is not set!")

    test_save_path = os.path.join(cf.training_info_path, "Test")

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
    model_files = [f for f in os.listdir(cf.training_info_path + 'save_model/') if
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

            model_path = os.path.join(cf.training_info_path, 'save_model', model)

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

    # acc and loss 曲线
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

    history_file_path = os.path.join(cf.training_info_path, f'Training_Information/training_history.csv')

    x_test, y_test, time_preread_indices, window_indices = load_tfrecord_to_list(
        cf.data_path + "processed_data/data_contact_test.tfrecord")
    x_test_tensor = tf.convert_to_tensor(x_test, dtype=tf.float32)
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

    # Retrieve and sort all files starting with "model_" in the directory;
    model_files = [f for f in os.listdir(cf.training_info_path + 'save_model/') if
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

            model_path = os.path.join(cf.training_info_path, 'save_model', model)

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
            failure_time_preread_indices = [time_preread_indices[i] for i in failure_indices]
            failure_window_indices = [window_indices[i] for i in failure_indices]
            failure_labels = [y_test[i] for i in failure_indices]
            failure_predicted_labels = [y_pred[i] for i in failure_indices]
            failure_file_path = os.path.join(cf.training_info_path,
                                             f'Error_Data_Information/failure_indices_{model}.csv')

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

    training_time = (cf.end_time - cf.start_time) / 60

    with open(cf.training_info_path + 'training_info.txt', 'w') as file:
        file.write(f'参与训练的收拾序号为{cf.gesture}')
        file.write(f'数据集划分方式为{cf.tvt_select_mode}')
        file.write(f'训练时间: {training_time:.2f} minutes\n')
        file.write(f'训练数据选择数量: {cf.train_num} \n')
        file.write(f'测试数据位置: {cf.test_nums} \n')
        file.write(f'验证数据位置: {cf.val_nums}\n')
        file.write(f'训练数据位置: {cf.train_nums}\n')
        file.write(f'测试数据选择数量: {cf.test_num} \n')
        file.write(f'验证数据选择数量: {cf.val_num} \n')
        file.write(f'窗口大小: {cf.window_size}\n')
        file.write(f'窗口步长: {cf.step_size}\n')
        file.write(f'小窗口大小: {cf.step_size_little}\n')
        file.write(f'小窗口步长: {cf.window_size_little}\n')
        file.write(f'训练次数: {cf.epochs}\n')
        file.write(f'缩放系数: {cf.scaling}\n')
        file.write(f'所调用的模型: {cf.model_name}\n')

    print(f"Total training time: {training_time:.2f} minutes")

    print("训练完成！")

    print(f"训练历史记录已保存到: {history_file_path}")

    test_loss, test_accuracy = cf.model.evaluate(test_dataset, verbose=2)

    print(f'\nTest accuracy: {test_accuracy * 100:.2f}%')

    print(f'\nTest loss: {test_loss:.4f}')

