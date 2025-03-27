#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/3/27 11:55
# @Author : Jason.LI
# @File : main.py
# @Software: PyCharm

from base_config.config import GlobalConfig
from models.LiteSTFNet.litestfnet import litestfnet_model_creat
from gesture_recognition.base.train_process import one_model_train
from gesture_recognition.base.dataset import database_create,tfrecord_connect

if __name__ == '__main__':

    cf = GlobalConfig()
    cf.config_init()
    cf.display_config()

    database_create(
        train_indices       = cf.train_indices,
        test_indices        = cf.test_indices,
        val_indices         = cf.val_indices,
        remaining_numbers   = cf.remaining_numbers,
        gesture_sequence    = cf.gesture_sequence,
        path_to_use_data    = cf.path_to_use_data,
        once_read_time      = cf.once_read_time,
        sample_rate         = cf.sample_rate,
        window_size         = cf.window_size,
        step_size           = cf.step_size,
        window_size_little  = cf.window_size_little,
        step_size_little    = cf.step_size_little,
    )

    tfrecord_connect(
        gesture_sequence    = cf.gesture_sequence,
        path_to_use_data    = cf.path_to_use_data,
        feature_shape       = cf.feature_shape
    )

    model,model_name = litestfnet_model_creat(
        feature_shape = cf.feature_shape,
        gesture_num   = cf.gesture_num
    )

    one_model_train(
        model       = model,
        config      = cf,
        model_name  = model_name
    )

    cf.display_config()