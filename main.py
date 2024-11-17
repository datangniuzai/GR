# -*- coding: utf-8 -*-
# @Time : 2024/6/23 23:20
# @Author : 李 嘉 轩
# @File : script.py
# @Software: PyCharm

import config as cf
import data_reading
from data_reading import online_data_read
from feature_extraction import online_data_contact

if __name__ == '__main__':

    cf.folder_path = "data/"
    cf.config_read()
    online_data_read()