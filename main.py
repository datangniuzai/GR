# -*- coding: utf-8 -*-
# @Time : 2024/6/23 23:20
# @Author : 李 嘉 轩
# @File : script.py
# @Software: PyCharm

import time
import config as cf
from dataset import online_dataset,dataset_connect
from model_file import TCCNN_model_creat
from train_process import model_train,Plot_matrix,Plot_loos_acc_matrix_test


if __name__ == '__main__':
    cf.config_read()
    # cf.start_time = time.time()
    # online_dataset()
    # dataset_connect()
    cf.model = TCCNN_model_creat()
    # cf.end_time = time.time()
    Plot_loos_acc_matrix_test()