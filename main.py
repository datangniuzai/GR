#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2024/11/14 19:48
# @Author : 李 嘉 轩
# @team member : 赵雨新
# @File : script.py
# @Software: PyCharm Vscode

import time
import config as cf
from dataset import online_dataset,dataset_connect
from model_file import creat_model
from train_process import model_train,Plot_matrix,Plot_loos_acc_matrix_test


if __name__ == '__main__':
    cf.config_read()
    # cf.start_time = time.time()
    #online_dataset()
    #dataset_connect()
    cf.model = creat_model()
    model_train()
    # cf.end_time = time.time()
    Plot_matrix()