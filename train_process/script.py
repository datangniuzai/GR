#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2024/11/14 20:45
# @Author : 李 嘉 轩
# @File : script.py
# @Software: PyCharm

import os
import datetime
import config as cf
import numpy as np
import tensorflow as tf
from data_reading import read_tfrecord

def model_train(self):

    # 获取当前时间
    current_time = datetime.datetime.now()
    # 格式化当前时间
    folder_name = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    # 创建主文件夹
    main_folder_path = os.path.join(self.folder_path, folder_name)
    os.makedirs(main_folder_path, exist_ok=True)
    # 创建子文件夹
    picture_folder_path = os.path.join(main_folder_path, "Picture")
    model_folder_path = os.path.join(main_folder_path, "save_model")
    errordata_information = os.path.join(main_folder_path, "Error_Data_Information")
    os.makedirs(picture_folder_path, exist_ok=True)
    os.makedirs(model_folder_path, exist_ok=True)
    os.makedirs(errordata_information, exist_ok=True)
    # 获取保存地址
    self.save_path = self.folder_path+folder_name+"/"
    # 获取GPU使用可行性
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    # 预设sgd优化器，以便后续调整学习率
    sgd_optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
    # 定义一个自定义回调函数来更换优化器
    class ChangeOptimizerCallback(tf.keras.callbacks.Callback):
        def __init__(self, sgd_optimizer, switch_epoch, learning_rate):
            super(ChangeOptimizerCallback, self).__init__()
            self.sgd_optimizer = sgd_optimizer
            self.switch_epoch = switch_epoch
            self.learning_rate = learning_rate
        def on_epoch_begin(self, epoch, logs=None):
            if epoch == self.switch_epoch:
                self.model.optimizer = self.sgd_optimizer
                tf.keras.backend.set_value(self.model.optimizer.learning_rate, self.learning_rate)
                print(f"Optimizer changed to {self.sgd_optimizer} at epoch {epoch}")
    # 创建回调函数
    change_optimizer_callback = ChangeOptimizerCallback(sgd_optimizer, switch_epoch=self.epochs-5,learning_rate=0.001)
    model_checkpoint = ModelCheckpoint(
        self.save_path + f'save_model/model_' + '{epoch:02d}.keras',
        save_weights_only=False, save_best_only=False, verbose=1)
    # 读取数据集的数据
    x_val, y_val, x_p , x_t = self.read_tfrecord(self.folder_path + "data/data_contact_val.tfrecord")
    x_train, y_train,v_p , v_t = self.read_tfrecord(self.folder_path + "data/data_contact_train.tfrecord")
    # 将数据集转换为 TensorFlow Dataset 对象
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(len(x_train)).batch(64)
    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(16)
    # 训练模型
    self.history = self.model.fit(train_dataset, validation_data=val_dataset, epochs=self.epochs,
                        callbacks=[model_checkpoint, change_optimizer_callback])

def Plot_matrix(data_path,model_path):
    X_test, y_test,x_p,x_t = read_tfrecord(data_path)
    # 测试集数据转化为张量
    X_test_tensor = tf.convert_to_tensor(X_test, dtype=tf.float32)
    model.load_weights(model_path)
    # 预测测试集
    y_pred_prob = model.predict(X_test_tensor)
    y_pred = np.argmax(y_pred_prob, axis=1)
    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    # 创建热图
    cm = confusion_matrix(y_test, y_pred)
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_perc, annot=True, cmap="YlGnBu", fmt=".1f", linewidths=.5, square=True,
                annot_kws={"fontsize": 12})
    plt.xlabel('Predicted label', fontsize=14)
    plt.ylabel('True label', fontsize=14)
    plt.title(f'Confusion Matrix (Accuracy: {accuracy * 100:.2f}%)', fontsize=16)
    plt.show()
def Plot_loosacc_everyepoch_matrix(self):
    # 读取测试集数据
    X_test, y_test, time_preread_indices, window_indices = self.read_tfrecord(
        self.folder_path + "data/data_contact_test.tfrecord")
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(32)
    # 测试集数据转化为张量
    X_test_tensor = tf.convert_to_tensor(X_test, dtype=tf.float32)
    # 绘制训练和验证的准确率曲线
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(self.history.history['accuracy'])
    plt.plot(self.history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # 绘制训练和验证的损失曲线
    plt.subplot(1, 2, 2)
    plt.plot(self.history.history['loss'])
    plt.plot(self.history.history['val_loss'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(self.save_path + f'picture/loss_acc.svg', format='svg')
    plt.close()
    # 获取所有保存的模型文件
    model_files = [f for f in os.listdir(self.save_path + 'save_model/') if
                   f.startswith(f'model_')]
    model_files.sort()  # 确保按顺序加载
    # 定义文件路径
    history_file_path = os.path.join(self.save_path, f'training_history.csv')
    with open(history_file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # 写入表头
        header = ['epoch', 'loss', 'accuracy', 'val_loss', 'val_accuracy', 'test_accuracy']
        for i in range(self.gesture_num):
            header.append(f'recall_gesture_{i}')
        writer.writerow(header)
        # 写入每个epoch的数据
        for epoch, model in enumerate(model_files):
            model_path = os.path.join(self.save_path, 'save_model', model)
            # 加载权重
            self.model.load_weights(model_path)
            # 预测测试集
            y_pred_prob = self.model.predict(X_test_tensor)
            y_pred = np.argmax(y_pred_prob, axis=1)
            # 计算准确率
            accuracy = accuracy_score(y_test, y_pred)
            # 计算召回率
            recall = recall_score(y_test, y_pred, average=None)
            # 创建热图
            cm = confusion_matrix(y_test, y_pred)
            cm_sum = np.sum(cm, axis=1, keepdims=True)
            cm_perc = cm / cm_sum.astype(float) * 100
            # 创建注释文本
            annot = np.zeros_like(cm_perc, dtype=object)
            for i in range(cm_perc.shape[0]):
                for j in range(cm_perc.shape[1]):
                    if cm_perc[i, j] != 0:
                        annot[i, j] = f"{cm_perc[i, j]:.1f}"
                    else:
                        annot[i, j] = ""
            plt.figure(figsize=(15, 12))
            sns.heatmap(cm_perc, annot=annot, cmap="YlGnBu", fmt='', linewidths=1, square=True,
                        annot_kws={"fontsize": 12})
            plt.xlabel('Predicted label', fontsize=14)
            plt.ylabel('True label', fontsize=14)
            plt.title(f'Accuracy: {accuracy * 100:.2f}%', fontsize=16)
            # 保存热图
            heatmap_path = os.path.join(self.save_path, f'picture/confusion_matrix_{model}.svg')
            plt.savefig(heatmap_path, format='svg')
            plt.close()
            # 记录预测失败的点的 window_index、time_preread_index、label 和 predicted_label
            failure_indices = np.where(y_pred != y_test)[0]
            failure_time_preread_indices = [time_preread_indices[i] for i in failure_indices]
            failure_window_indices = [window_indices[i] for i in failure_indices]
            failure_labels = [y_test[i] for i in failure_indices]
            failure_predicted_labels = [y_pred[i] for i in failure_indices]
            failure_file_path = os.path.join(self.save_path, f'Error_Data_Information/failure_indices_{model}.csv')
            with open(failure_file_path, 'w', newline='') as failure_file:
                failure_writer = csv.writer(failure_file)
                failure_writer.writerow(['window_index', 'time_preread_index', 'label', 'predicted_label'])
                for window_index, time_preread_index, label, predicted_label in zip(failure_window_indices,
                                                                                    failure_time_preread_indices,
                                                                                    failure_labels,
                                                                                    failure_predicted_labels):
                    failure_writer.writerow([window_index, time_preread_index, label, predicted_label])
            # 写入每个epoch的数据
            row = [
                epoch + 1,
                self.history.history['loss'][epoch],
                self.history.history['accuracy'][epoch],
                self.history.history['val_loss'][epoch],
                self.history.history['val_accuracy'][epoch],
                accuracy
            ]
            row.extend(recall)
            writer.writerow(row)
    model_name = self.model.name
    # 计算训练时间
    training_time = (self.end_time - self.start_time) / 60
    # 保存训练时间和选择的通道
    with open(self.save_path + 'training_info.txt', 'w') as file:
        if self.selected_channels is None or not self.selected_channels.any():
            pass
        else:
            file.write(f'selected_channels: {self.selected_channels}\n')
        file.write(f'训练时间: {training_time:.2f} minutes\n')
        file.write(f'训练数据选择数量: {self.train_num} \n')
        file.write(f'训练数据位置: {self.train_nums}\n')
        file.write(f'测试数据选择数量: {self.test_num} \n')
        file.write(f'测试数据位置: {self.test_nums} \n')
        file.write(f'验证数据选择数量: {self.val_num} \n')
        file.write(f'验证数据位置: {self.val_nums}\n')
        file.write(f'窗口大小: {self.window_size}\n')
        file.write(f'窗口步长: {self.step_size}\n')
        file.write(f'小窗口大小: {self.step_size_little}\n')
        file.write(f'小窗口步长: {self.window_size_little}\n')
        file.write(f'训练次数: {self.epochs}\n')
        file.write(f'缩放系数: {self.scaling}\n')
        file.write(f'所调用的模型: {self.model_name}\n')  # 写入所调用的模型名称
    print(f"Total training time: {training_time:.2f} minutes")
    # 打印训练结果摘要
    print("训练完成！")
    print(f"训练历史记录已保存到: {history_file_path}")
    test_loss, test_accuracy = self.model.evaluate(test_dataset, verbose=2)
    print(f'\nTest accuracy: {test_accuracy * 100:.2f}%')  # :.2f 表示保留两位小数
    print(f'\nTest loss: {test_loss:.4f}')  # 保留四位小数来显示损失值