# -*- coding: utf-8 -*-
# @Time : 2024/6/23 23:20
# @Author : JIAXUAN LI
# @File : script.py
# @Software: PyCharm


import config as cf
from data_reading import sEMG_data_read_save



if __name__ == '__main__':
    cf.config_read()
    sEMG_data_read_save()
    # cf.start_time = time.time()
    # online_dataset()
    # dataset_connect()
    # cf.model = TCCNN_model_creat()
    # cf.end_time = time.time()
    # Plot_loos_acc_matrix_test()