#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2024/11/14 19:28
# @Author : Jason.LI
# @File : __init__.py.py
# @Software: PyCharm

import datetime
import json
import os
import random
from pathlib import Path
from typing import List, Tuple

from gesture_recognition.script.data_reading import sEMG_data_read_save


# configs
num_channels        = 64          # 通道数
sample_rate         = 2000        # 采样率
collector_number    = None        # 采集器编号

data_path           = None        # 文件夹地址
feature_shape       = None        # 输入特征形状
training_info_path  = None        # 本次训练所生成的目录
train_info_csv_path = None        # 训练过程数据保存地址
test_info_csv_path  = None        # 测试集验证细节

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

start_train_time    = None        # 模型训练开始的时间
end_train_time      = None        # 模型训练结束的时间

model               = None        # 模型
model_path          = None        # 模型地址
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

    project_root = find_project_root()
    os.chdir(project_root)

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

        sEMG_data_read_save()

    elif set_pattern == '2':
        # data process
        global window_size, step_size, window_size_little, step_size_little, epochs, model_name
        global train_num, val_num, test_num, train_nums, val_nums, test_nums, feature_shape

        # data process parameters
        dpp = config['data_process_parameters']

        data_path = dpp['data_path'] + "/"

        window_size = dpp['window_size']
        step_size = dpp['step_size']

        window_size_little = dpp['window_size_little']
        step_size_little = dpp['step_size_little']

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
                train_nums,test_nums,val_nums, remaining_numbers= split_data(turn_read_sum, train_num, test_num, val_num)

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
    os.makedirs(os.path.join(folder_name, 'all_train_info'))


    folder_path = os.path.abspath(folder_name)

    print(f"创建了数据集的文件夹，所在位置为： {folder_path}")
    folder_basename = os.path.basename(folder_path)
    data_path = folder_basename + "/"

def split_data(_turn_read_sum: int, _train_num: int, _test_num: int, _val_num: int) \
        -> Tuple[List[int], List[int], List[int], List[int]]:
    """
    Split the dataset into training, testing, validation sets and return any remaining numbers.

    Parameters:
    - turn_read_sum: The total number of samples in the dataset (int).
    - train_num: The number of samples in the training set (int).
    - test_num: The number of samples in the testing set (int).
    - val_num: The number of samples in the validation set (int).

    Returns:
    - A tuple of four lists:
      - train_nums: A list of sample indices for the training set (List[int]).
      - test_nums: A list of sample indices for the testing set (List[int]).
      - val_nums: A list of sample indices for the validation set (List[int]).
      - remaining_nums: A list of remaining sample indices (List[int]).
    """
    all_numbers = list(range(1, _turn_read_sum + 1))  # List of all sample indices

    _train_nums = random.sample(all_numbers, _train_num)

    _remaining_numbers = [num for num in all_numbers if num not in _train_nums]
    _test_nums = random.sample(_remaining_numbers, _test_num)


    _remaining_numbers = [num for num in _remaining_numbers if num not in _test_nums]
    _val_nums = random.sample(_remaining_numbers, _val_num)
    _remaining_numbers = [num for num in _remaining_numbers if num not in _val_nums]

    return _train_nums, _test_nums, _val_nums, _remaining_numbers

def find_project_root(start_path=None):
    if start_path is None:
        start_path = Path(__file__).resolve()

    while start_path != start_path.parent:
        if (start_path / 'README.md').exists():
            return start_path
        start_path = start_path.parent

    raise FileNotFoundError("Project root directory not found. "
                            "Please ensure the current directory contains a README.md file, "
                            "or modify the root directory search criteria as needed.")



if __name__ == '__main__':
    config_read()