#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2024/11/14 19:28
# @Author : 李 嘉 轩
# @File : __init__.py.py
# @Software: PyCharm

import os
import datetime
import json
import random

# configs
num_channels        = 64          # 通道数
sample_rate         = 2000        # 采样率
collector_number    = None        # 采集器编号

feature_shape       = None        # 输入特征形状
training_info_path  = None        # 模型等信息保存地址

data_path           = None        # 文件夹地址

tvt_select_mode     = None        # 数据集选择方式

train_num           = None        # 训练集数量
test_num            = None        # 测试集数量
val_num             = None        # 验证集数量

train_nums          = None        # 训练集位置
test_nums           = None        # 测试集位置
val_nums            = None        # 验证集位置

gesture_num         = None        # 手势数量
gesture             = None        # 手势序号

gesture_rest        = None        # 手势之间的休息时长
action_rest         = None        # 动作之间的休息时长

turn_read_sum       = None        # 读取总次数
time_preread        = None        # 每次读取的时长

window_size         = None        # 窗口大小
step_size           = None        # 窗口步长
window_size_little  = None        # 小窗口大小
step_size_little    = None        # 小窗口步长

epochs              = None        # 网络循环次数

scaling             = None        # 数据缩放系数

start_time          = None        # 数据集开始整理的时间
end_time            = None        # 模型训练结束整理的时间

model               = None        # 模型
model_name          = None        # 模型名称
history             = None        # 训练历史


def get_data_set_model():
    global tvt_select_mode
    while True:
        data_set_model = input("请选择数据集划分模式：1.随机；2.指定（请输入选项）：")
        if data_set_model in ['1', '2']:
            tvt_select_mode = "随机" if data_set_model == '1' else "指定"
            return data_set_model
        else:
            print("⚠️ 输入无效！请输入'1'或'2'以选择数据集划分模式。\n")

def pattern_set():
    while True:
        pattern_mode = input("请选择使用模式：1.数据读取；2.数据处理（请输入选项）:")
        if pattern_mode in ['1', '2']:
            return pattern_mode
        else:
            print("⚠️ 输入无效！请输入'1'或'2'以选择使用模式。\n")

def config_read():

    global gesture_num, gesture, turn_read_sum, time_preread , data_path ,collector_number

    global action_rest, gesture_rest

    with open('config/training_config.json', 'r', encoding='utf-8') as f:
        config = json.load(f)

    set_pattern = pattern_set()

    if set_pattern == '1':

        # data reading parameters
        drp = config['data_reading_parameters']

        collector_number = drp['collector_number'] + 8079

        turn_read_sum = drp["gesture_read_count_per_instance"]
        time_preread = drp["read_duration_per_instance"]
        action_rest = drp["action_rest_duration"]
        gesture_rest = drp["gesture_rest_duration"]

        gesture = drp["gesture_read_sequence"]
        gesture_num = len(gesture)

        data_folder_create()

        print("|     采集器的编号    |", collector_number - 8079)
        print("|     手势动作序号    |", gesture)
        print("|     手势动作数量    |", gesture_num)
        print("|     单次读取时间    |", time_preread,"(s)")
        print("|     手势重复次数    |", turn_read_sum)
        print("|     动作之间休息    |", action_rest,"(s)")
        print("|     手势之间休息    |", gesture_rest,"(s)")

    elif set_pattern == '2':
        # data process
        global window_size, step_size, window_size_little, step_size_little, epochs, scaling, model_name
        global train_num, val_num, test_num, train_nums, val_nums, test_nums, feature_shape

        # data process parameters
        dpp = config['data_process_parameters']

        data_path = dpp['data_path'] + "/"

        window_size = dpp['window_size']
        step_size = dpp['step_size']

        window_size_little = dpp['window_size_little']
        step_size_little = dpp['step_size_little']

        scaling = dpp['scaling']

        feature_shape = dpp['feature_shape']

        gesture = dpp['gesture_train_sequence']
        gesture_num = len(gesture)

        # model train parameters
        mtp = config['model_train_parameters']

        epochs = mtp['epochs']
        model_name = mtp['model_name']

        # read vol info
        info_file_path = str(data_path) + "vol_exp_info.json"
        with open(info_file_path, 'r', encoding='utf-8') as f:
            info = json.load(f)

        identifier = info['identifier']

        experiment_info = info['experiment_info']

        turn_read_sum = experiment_info['gesture_read_count_per_instance']
        time_preread = experiment_info['read_duration_per_instance']

        # train, val, test data set
        data_set_model = get_data_set_model()

        if data_set_model == '1':
            # random
            train_num = dpp["random_data_set_mode"]['train_num']
            val_num = dpp["random_data_set_mode"]['val_num']
            test_num = dpp["random_data_set_mode"]['test_num']

            if train_num + val_num + test_num > turn_read_sum:
                raise ValueError(f"调用{train_num + val_num + test_num}次所采集数据，但总采集次数仅有{turn_read_sum}次！")

            else:
                all_numbers = list(range(1, turn_read_sum + 1))
                train_nums = random.sample(all_numbers, train_num)

                remaining_numbers = [num for num in all_numbers if num not in train_nums]
                test_nums = random.sample(remaining_numbers, test_num)

                remaining_numbers = [num for num in remaining_numbers if num not in test_nums]
                val_nums = random.sample(remaining_numbers, val_num)

                remaining_numbers = [num for num in remaining_numbers if num not in val_nums]

        else:
            # special
            train_nums = dpp["specify_data_set_mode"]['train_nums']
            val_nums = dpp["specify_data_set_mode"]['val_nums']
            test_nums = dpp["specify_data_set_mode"]['test_nums']

            train_num = len(train_nums)
            val_num = len(val_nums)
            test_num = len(test_nums)

            if train_num + val_num + test_num > turn_read_sum:
                raise ValueError(f"调用{train_num + val_num + test_num}次所采集数据，但总采集次数仅有{turn_read_sum}次！")

            remaining_numbers = list(set(range(1, turn_read_sum+1)) - set(train_nums + val_nums + test_nums))

        print("|     数据集的编号    |", identifier)
        print("|     手势动作序号    |", gesture)
        print("|     手势动作数量    |", gesture_num)
        print("|     一级窗口大小    |", window_size)
        print("|     一级窗口步长    |", step_size)
        print("|     二级窗口大小    |", window_size_little)
        print("|     二级窗口步长    |", step_size_little)
        print("|     模型迭代次数    |", epochs)
        print("|     数据缩放系数    |", scaling)
        print("|     训练集的位置    |", train_nums)
        print("|     验证集的位置    |", val_nums)
        print("|     测试集的位置    |", test_nums)

        if remaining_numbers:
            print("|     未使用的位置    |", remaining_numbers)
        else:
            print("|     未使用的位置    |  无")

def data_folder_create():

    global data_path

    current_time = datetime.datetime.now()
    folder_name = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(folder_name)
    os.makedirs(os.path.join(folder_name, 'processed_data'))
    os.makedirs(os.path.join(folder_name, 'original_data'))
    os.makedirs(os.path.join(folder_name, 'picture'))
    os.makedirs(os.path.join(folder_name, 'train_info'))

    folder_path = os.path.abspath(folder_name)

    print(f"创建了数据集的文件夹，所在位置为： {folder_path}")
    folder_basename = os.path.basename(folder_path)
    data_path = folder_basename + "/"

if __name__ == '__main__':
    config_read()