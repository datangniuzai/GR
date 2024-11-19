# -*- coding: utf-8 -*-
# @Time : 2024/6/23 23:20
# @Author : 李 嘉 轩
# @File : script.py
# @Software: PyCharm

import config as cf
from feature_extraction import online_dataset,dataset_connect

if __name__ == '__main__':
    cf.config_read()
    online_dataset()
    dataset_connect()
