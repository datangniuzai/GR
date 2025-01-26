# -*- coding: utf-8 -*-
# @Time : 2024/1/11 23:20
# @Author : Yuxin Zhao
# @File : train_process.py
# @Software: Vscode

import os
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import time
import config as cf
from dataset import database_create,tfrecord_connect
from model_file import creat_model
from train_process import model_train,Plot_matrix


if __name__ == '__main__':
    cf.config_read()
    cf.start_time = time.time()
    database_create()
    tfrecord_connect()
    cf.model = creat_model()
    model_train()
    cf.end_time = time.time()
    Plot_matrix()