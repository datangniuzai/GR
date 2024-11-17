#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2024/11/14 19:28
# @Author : 李 嘉 轩
# @File : __init__.py.py
# @Software: PyCharm

import random
import json

# configs
num_channels        = 64          # 通道数
sample_rate         = 2000        # 采样率

feature_shape       = None        # 输入特征形状
save_path           = None        # 模型等信息保存地址

folder_path         = None        # 文件夹地址

train_num           = None        # 训练集数量
test_num            = None        # 测试集数量
val_num             = None        # 验证集数量

train_nums          = None        # 训练集位置
test_nums           = None        # 测试集位置
val_nums            = None        # 验证集位置

gesture_num         = None        # 手势数量
gesture             = None        # 手势序号

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



def config_read():
    global gesture_num, gesture, turn_read_sum, time_preread , folder_path
    global train_num, val_num, test_num
    global train_nums,val_nums,test_nums
    global window_size,step_size,window_size_little,step_size_little,epochs,scaling

    # read vol info
    volunteer_info = {}
    info_file_path = folder_path + "vol&ges_info.txt"
    with open(info_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if line.strip():
                key, value = line.strip().split(': ', 1)
                if value.startswith('[') and value.endswith(']'):
                    value = eval(value)
                elif value.isdigit():
                    value = int(value)
                volunteer_info[key] = value
    gesture = volunteer_info.get('手势序号')
    gesture_num = len(gesture)
    turn_read_sum = int(volunteer_info.get('一个手势的读取次数'))
    time_preread = int(volunteer_info.get('每次读取的时间长度'))

    # read model and training info
    with open('config/training_config.json', 'r', encoding='utf-8') as f:
        config = json.load(f)

    parameters = config['used_parameters']

    window_size = parameters['window_size']
    step_size = parameters['step_size']
    window_size_little = parameters['window_size_little']
    step_size_little= parameters['step_size_little']
    epochs = parameters['epochs']
    scaling = parameters['scaling']

    used_params = config.get('used_parameters', {})

    if 'train_num' in used_params:
        train_num = used_params['train_num']
        val_num = used_params['val_num']
        test_num = used_params['test_num']

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
        train_nums = used_params['train_nums']
        val_nums = used_params['val_nums']
        test_nums = used_params['test_nums']

        train_num = len(train_nums)
        val_num = len(val_nums)
        test_num = len(test_nums)

        if train_num + val_num + test_num > turn_read_sum:
            raise ValueError(f"调用{train_num + val_num + test_num}次所采集数据，但总采集次数仅有{turn_read_sum}次！")

        remaining_numbers = list(set(range(1, turn_read_sum + 1)) - set(train_nums + val_nums + test_nums))

    print("|     志愿者年龄    |", volunteer_info.get('年龄'))
    print("|     志愿者性别    |", volunteer_info.get('性别'))
    print("|     志愿者手臂    |", volunteer_info.get('测量手臂'))
    print("|     手势的序号    |", gesture)
    print("|     手势的数量    |", gesture_num)
    print("|     窗口的大小    |", window_size)
    print("|     窗口的步长    |", step_size)
    print("|     小窗口大小    |", window_size_little)
    print("|     小窗口步长    |", step_size_little)
    print("|      epochs     |", epochs)
    print("|     缩放的系数    |", scaling)
    print("|     训练集位置    |", train_nums)
    print("|     验证集位置    |", val_nums)
    print("|     测试集位置    |", test_nums)

    if remaining_numbers:
        print("|     未使用位置    |", remaining_numbers)
    else:
        print("|     未使用位置    |  无")
