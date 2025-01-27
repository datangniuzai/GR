# -*- coding: utf-8 -*-
# @Time : 2024/1/11 23:20
# @Author : Yuxin Zhao
# @File : train_process.py
# @Software: Vscode

import os
import time
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import config as cf
from model_file import creat_model
from train_process import model_train,plot_confusion_matrix
from dataset import database_create,tfrecord_connect

if __name__ == '__main__':
    cf.config_read()
    cf.start_time = time.time()
    # database_create()
    print("data creation over")
    # tfrecord_connect()
    print("data connection over")
    cf.model = creat_model()
    model_train()
    cf.end_time = time.time()
    plot_confusion_matrix()