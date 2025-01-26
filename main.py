# -*- coding: utf-8 -*-
# @Time : 2024/1/11 23:20
# @Author : 赵 雨 新
# @File : script.py
# @Software: Vscode

import time
import config as cf
from dataset import online_dataset,dataset_connect
from model_file import creat_model
from train_process import model_train,Plot_matrix,Plot_loos_acc_matrix_test
import tensorflow as tf


if __name__ == '__main__':
    cf.config_read()
    cf.start_time = time.time()
    online_dataset()
    dataset_connect()
    cf.model = creat_model()
    model_train()
    cf.end_time = time.time()
    Plot_matrix()