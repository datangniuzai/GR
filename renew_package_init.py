#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2024/11/17 05:18
# @Author : 李 嘉 轩
# @File : renew_package_init.py
# @Software: PyCharm

import os
import ast

def generate_init_file(package_dir):
    init_file = os.path.join(package_dir, '__init__.py')
    content_lines = []

    for file in os.listdir(package_dir):
        if file.endswith('.py') and file != '__init__.py':
            file_path = os.path.join(package_dir, file)
            module_name = file[:-3]

            with open(file_path, 'r', encoding='utf-8') as f:
                node = ast.parse(f.read())
                for n in node.body:
                    if isinstance(n, ast.FunctionDef):  # 检查是否是函数
                        content_lines.append(f'from .{module_name} import {n.name}')
                    elif isinstance(n, ast.ClassDef):  # 检查是否是类
                        content_lines.append(f'from .{module_name} import {n.name}')

    with open(init_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(content_lines))
        f.write('\n')

# 替换 package_dir 为你的包路径
package_dir = "my_models"
generate_init_file(package_dir)
