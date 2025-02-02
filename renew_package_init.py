#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2024/11/17 05:18
# @Author : JIAXUAN LI
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
                    if isinstance(n, ast.FunctionDef):
                        content_lines.append(f'from .{module_name} import {n.name}')
                    elif isinstance(n, ast.ClassDef):
                        content_lines.append(f'from .{module_name} import {n.name}')

    with open(init_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(content_lines))
        f.write('\n')
if __name__ == '__main__':
    # 替换 package_dir 为你的包路径
    package_dir = "train_process"
    generate_init_file(package_dir)
