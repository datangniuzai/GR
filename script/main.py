# -*- coding: utf-8 -*-
# @Time : 2024/1/11 23:20
# @Author : Yuxin Zhao
# @File : train_process.py
# @Software: Vscode

import os
import time
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import config as cf
from model_file import tccnn_model_creat
from train_process import one_model_train, k_fold_cross_validation
from dataset import database_create,tfrecord_connect

if __name__ == '__main__':
    cf.config_read()
    k_fold_cross_validation(2)


    cf.start_time = time.time()
    # database_create()
    # tfrecord_connect()
    cf.model = tccnn_model_creat()
    one_model_train(model_name = "LiteSTFNet")
    cf.end_time = time.time()