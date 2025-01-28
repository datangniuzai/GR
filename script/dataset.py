#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2024/11/14 19:48
# @Author : Jiaxuan LI
# @File : filtering.py
# @Software: PyCharm

import os
from typing import Any, List, Tuple, Dict

import numpy as np
import pandas as pd
import tensorflow as tf

import config as cf
from filtering import bandpass_and_notch_filter


# ------------------------------ #
# Reset Adjacency Functionality  #
# ------------------------------ #

def build_one_adjacency():
    one_adjacency = np.array([
        [0,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0],
        [1,0,1,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0],
        [0,1,0,1,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1],
        [0,0,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
        [1,1,0,0,0,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [1,1,1,0,1,0,1,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,1,1,1,0,1,0,1,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,1,1,0,0,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,1,1,0,0,0,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,1,1,1,0,1,0,1,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,1,1,1,0,1,0,1,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,1,1,0,0,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,1,1,0,0,0,1,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,1,1,1,0,1,0,1,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,1,1,1,0,0,1,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,1,1,0,0,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,1,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,1,0,1,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,1,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,1,0,1,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,1,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,1,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,1,0,1,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,1,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,1,0,1,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,1,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,1,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,1,0,1,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,1,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,1,1,1,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,1,1,1,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,1,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,1,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,1,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,1,0,1,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,1,0,0,0,0,0],
        [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0],
        [1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0],
        [0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1],
        [0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],
                    ])
    return one_adjacency

def adj_to_bias(adj, nood):
    nb_graphs = adj.shape[0]
    mt = np.empty(adj.shape)
    for g in range(nb_graphs):
        mt[g] = np.eye(adj.shape[1])
        for _ in range(nood):
            mt[g] = np.matmul(mt[g], (adj[g] + np.eye(adj.shape[1])))
        for i in range(64):
            for j in range(64):
                if mt[g][i][j] > 0.0:
                    mt[g][i][j] = 1.0

    return -1e9 * (1.0 - mt)

def fill_new_adjacency_matrix(sampled_neighbors,num_graphs,num_nodes):
    new_adjacency_matrix = np.zeros((num_graphs,num_nodes, num_nodes), dtype=np.int32)

    for (graph_idx, node), neighbors in sampled_neighbors.items():
        for neighbor_idx in neighbors:
            if neighbor_idx < num_nodes:
                new_adjacency_matrix[graph_idx, node, neighbor_idx] = 1

    return new_adjacency_matrix

def sample_neighbors(adjacency_matrix, num_samples):
    sampled_neighbors = {}
    num_graphs, num_nodes, _ = adjacency_matrix.shape
    for graph_idx in range(num_graphs):
        graph_adjacency = adjacency_matrix[graph_idx]
        for node in range(num_nodes):
            neighbors = np.where(graph_adjacency[node] == 1)[0]
            num_neighbors = len(neighbors)
            sampled_indices = np.random.choice(num_neighbors, num_samples, replace=True)
            sampled_neighbors_idx = neighbors[sampled_indices]
            sampled_neighbors[(graph_idx, node)] = sampled_neighbors_idx
    return sampled_neighbors

def reset_adj(graph_count: int) -> Any:
    """
    Generate a bias-adjusted adjacency matrix for a specified number of graphs.

    Args:
        graph_count (int): The number of graphs to generate adjacency matrices for.

    Returns:
        Any: A bias-adjusted adjacency matrix suitable for use in graph-based machine learning models.
    """
    init_adj = np.array([build_one_adjacency()] * graph_count)

    sampled_adj = sample_neighbors(init_adj, 5)

    filled_adj = fill_new_adjacency_matrix(sampled_adj, graph_count, 64)

    bias_adj = adj_to_bias(filled_adj, 1)

    return bias_adj

# ------------------------------ #
#   Feature Extraction Function  #
# ------------------------------ #

def calc_td(data: np.ndarray) -> np.ndarray:
    """
    Extract only time-domain features, add small windows. Feature order: MAV, RMS, MSE, Zero-crossings, WAMP.

    :param data: Input data matrix with shape (num_channels, signal_length)
    :return: Extracted features with shape (num_windows, num_channels, 5)
    """
    signal_length, num_channels, = data.shape

    num_windows = (signal_length - cf.window_size_little) // cf.step_size_little + 1

    features = []

    willison_threshold = 20 / cf.scaling

    for i in range(num_windows):

        start = i * cf.step_size_little
        end = start + cf.window_size_little
        windowed_data = data[start:end, :]

        # Mean Absolute Value (MAV)
        mav = np.mean(np.abs(windowed_data), axis=0)
        # Root Mean Square (RMS)
        rms = np.sqrt(np.mean(windowed_data ** 2, axis=0))
        # Mean Squared Error (MSE)
        mse = np.mean((windowed_data - np.mean(windowed_data, axis=0, keepdims=True)) ** 2, axis=0)
        # Zero-crossings
        zero_crossings = np.sum(np.diff(np.sign(windowed_data), axis=0) != 0, axis=0)
        # Willison Amplitude (WAMP)
        willison_amplitudes = np.sum(np.abs(np.diff(windowed_data, axis=0)) > willison_threshold, axis=0)
        # Stack the features together
        feature = np.array([mav, rms, mse, zero_crossings, willison_amplitudes])

        features.append(feature)

    features = np.array(features)
    normalized_features = z_score_normalize_per_feature(features)

    return normalized_features

def z_score_normalize_per_feature(features: np.ndarray) -> np.ndarray:
    normalized_features = (features - np.mean(features, axis=(0, 2), keepdims=True)) / np.std(features, axis=(0, 1), keepdims=True)
    return normalized_features

def z_score_normalize_per_channel(features: np.ndarray) -> np.ndarray:
    normalized_features = (features - np.mean(features, axis=(0, 1), keepdims=True)) / np.std(features, axis=(0, 2), keepdims=True)
    return normalized_features

def z_score_normalize_per_window(features: np.ndarray) -> np.ndarray:
    normalized_features = (features - np.mean(features, axis=(1, 2), keepdims=True)) / np.std(features, axis=(1, 2), keepdims=True)
    return normalized_features

def min_max_normalize_per_feature(features: np.ndarray) -> np.ndarray:
    features_min = np.min(features, axis=(0, 2), keepdims=True)
    features_max = np.max(features, axis=(0, 2), keepdims=True)
    return (features - features_min) / (features_max - features_min)

def min_max_normalize_per_channel(features: np.ndarray) -> np.ndarray:
    features_min = np.min(features, axis=(0, 1), keepdims=True)
    features_max = np.max(features, axis=(0, 1), keepdims=True)
    return (features - features_min) / (features_max - features_min)

def min_max_normalize_per_window(features: np.ndarray) -> np.ndarray:
    features_min = np.min(features, axis=(1, 2), keepdims=True)
    features_max = np.max(features, axis=(1, 2), keepdims=True)
    return (features - features_min) / (features_max - features_min)

# ------------------------------ #
#   Tfrecord Build Function      #
# ------------------------------ #

def tfrecord_establish(df: np.ndarray, gesture_number: int, dataset_type: str):
    """
    General data processing function for feature extraction and saving for training, testing, and validation datasets.

    This function extracts features from input signals, processes them, and saves them as TensorFlow TFRecord files.

    :param df: Input signal data (shape: [num_channels, signal_length])
    :param gesture_number: Gesture identifier (integer)
    :param dataset_type: Type of dataset ('train'/'test'/'val')
    :return: The element_spec of the dataset
    """

    window_data_features = []
    window_data_labels = []
    window_data_time_preread_indexs = []
    window_data_window_indexs = []

    for read_time in range(1, cf.turn_read_sum + 1):
        if read_time in getattr(cf, f"{dataset_type}_nums"):
            single_acqui_data = df[(read_time - 1) * (cf.time_preread * cf.sample_rate):read_time * (
                        cf.time_preread * cf.sample_rate), :]
            single_acqui_data = bandpass_and_notch_filter(single_acqui_data)
            for j in range(0, single_acqui_data.shape[0] - cf.window_size + 1, cf.step_size):
                window_data = single_acqui_data[j:j + cf.window_size, :]
                window_data_features.append(calc_td(window_data))
                window_data_labels.append(gesture_number - 1)
                window_data_time_preread_indexs.append(read_time)
                window_data_window_indexs.append(j)

    graph_count = len(window_data_features)
    adjacencies = reset_adj(graph_count)

    window_data_feature_tensor = tf.convert_to_tensor(window_data_features, dtype=tf.float32)
    adjacency_tensor = tf.convert_to_tensor(adjacencies, dtype=tf.float32)
    label_tensor = tf.convert_to_tensor(window_data_labels, dtype=tf.uint8)
    time_preread_index_tensor = tf.convert_to_tensor(window_data_time_preread_indexs, dtype=tf.uint8)
    window_index_tensor = tf.convert_to_tensor(window_data_window_indexs, dtype=tf.uint8)

    dataset = tf.data.Dataset.from_tensor_slices(
        (window_data_feature_tensor, adjacency_tensor, label_tensor, time_preread_index_tensor, window_index_tensor))

    save_path = os.path.join(cf.data_path, "processed_data")
    os.makedirs(save_path, exist_ok=True)

    tfrecord_path = os.path.join(save_path, f"data_{gesture_number}_{dataset_type}.tfrecord")
    tfrecord_save(dataset, tfrecord_path)

    cf.feature_shape = window_data_features[1].shape

def tfrecord_connect():

    for dataset_type in ['train','test','val']:

        merged_dataset = None

        for gesture_number in cf.gesture:

            dataset= load_tfrecord_to_dataset(cf.data_path + f"processed_data/data_{gesture_number}_{dataset_type}.tfrecord")

            if merged_dataset is None:
                merged_dataset = dataset
            else:
                merged_dataset = merged_dataset.concatenate(dataset)

        connect_tfrecord_save_path = os.path.join(cf.data_path,f"processed_data/data_contact_{dataset_type}.tfrecord")

        tfrecord_save(merged_dataset,connect_tfrecord_save_path)

        print(f"[{dataset_type}] data has been merged and saved at [{connect_tfrecord_save_path}]")

    print("data connection over")

def tfrecord_save(dataset :tf.data.Dataset,tfrecord_save_path:str):
    """
    Save the dataset as a TFRecord file.

    Parameters:
    dataset (iterable): A dataset containing window data, labels, and generated adjacency matrices.
    tfrecord_save_path (str): The path where the TFRecord file will be saved.

    Functionality:
    Converts each item in the dataset (window data, labels, etc.) to `tf.train.Example` format and writes it to the specified TFRecord file.
    """
    with tf.io.TFRecordWriter(tfrecord_save_path) as writer:
        for window, adjacency, label,time_preread_index, window_index in dataset:
            feature = {
                'window': tf.train.Feature(float_list=tf.train.FloatList(value=window.numpy().flatten())),
                'adjacency': tf.train.Feature(float_list=tf.train.FloatList(value=adjacency.numpy().flatten())),
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label.numpy().item()])),
                'time_preread_index': tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[time_preread_index.numpy().item()])),
                'window_index': tf.train.Feature(int64_list=tf.train.Int64List(value=[window_index.numpy().item()])),
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())

# ----------------------------- #
#   Tfrecord Loading Function   #
# ----------------------------- #

def _parse_function(proto: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Parse each Example from the TFRecord file and adjust the data types and shapes.

    Parameters:
    proto (tf.Tensor): The input TFRecord data.

    Returns:
    Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]: Parsed data including window data, adjacency matrix, label, time preread index, and window index.
    """
    keys_to_features: Dict[str, tf.io.FixedLenFeature] = {
        'window': tf.io.FixedLenFeature(cf.feature_shape, tf.float32),
        'adjacency': tf.io.FixedLenFeature([64, 64], tf.float32),
        'label': tf.io.FixedLenFeature([1], tf.int64),
        'time_preread_index': tf.io.FixedLenFeature([1], tf.int64),
        'window_index': tf.io.FixedLenFeature([1], tf.int64)
    }

    data = tf.io.parse_single_example(proto, keys_to_features)

    data['label'] = tf.cast(data['label'], tf.uint8)
    data['time_preread_index'] = tf.cast(data['time_preread_index'], tf.uint8)
    data['window_index'] = tf.cast(data['window_index'], tf.uint8)

    return data["window"], data['adjacency'], data['label'], data['time_preread_index'], data['window_index']

def load_tfrecord_to_dataset(tfrecord_path: str) -> tf.data.Dataset:
    """
    Load data from a TFRecord file.

    Parameters:
    tfrecord_path (str): Path to the TFRecord file.

    Returns:
    tf.data.Dataset: A TensorFlow dataset containing window data, labels, time preread indices, and window indices.
    """

    dataset = tf.data.TFRecordDataset(tfrecord_path)

    dataset = dataset.map(_parse_function)

    return dataset

def load_tfrecord_to_list(tfrecord_path: str) -> Tuple[List[np.ndarray], List[np.ndarray], List[int], List[int], List[int]]:
    """
    Load the TFRecord file and return the data as lists.

    Parameters:
    tfrecord_path (str): Path to the TFRecord file.

    Returns:
    Tuple[List[np.ndarray], List[np.ndarray], List[int], List[int], List[int]]:
        A tuple containing:
        - window_datas (List[np.ndarray]): List of window data arrays.
        - adjacencies (List[np.ndarray]): List of adjacency matrix arrays.
        - labels (List[int]): List of label integers.
        - time_preread_indices (List[int]): List of time preread indices as integers.
        - window_indices (List[int]): List of window indices as integers.
    """
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(_parse_function)

    window_datas: List[np.ndarray] = []
    adjacencies: List[np.ndarray] = []
    labels: List[int] = []
    time_preread_indices: List[int] = []
    window_indices: List[int] = []

    for window_data, adjacency, label, time_preread_index, window_index in dataset:
        window_datas.append(window_data.numpy())
        adjacencies.append(adjacency.numpy())
        labels.append(label.numpy())
        time_preread_indices.append(time_preread_index.numpy())
        window_indices.append(window_index.numpy())

    return window_datas, adjacencies, labels, time_preread_indices, window_indices

def load_tfrecord_to_tensor(tfrecord_path: str) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Load the TFRecord file and return window data, adjacency matrix, labels,
    time preread indices, and window indices as Tensors.

    Parameters:
    tfrecord_path (str): Path to the TFRecord file.

    Returns:
    Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]: A tuple containing:
        - window_datas (tf.Tensor): Tensor containing window data.
        - adjacencies (tf.Tensor): Tensor containing adjacency matrices.
        - labels (tf.Tensor): Tensor containing labels.
        - time_preread_indices (tf.Tensor): Tensor containing time preread indices.
        - window_indices (tf.Tensor): Tensor containing window indices.
    """

    dataset = tf.data.TFRecordDataset(tfrecord_path)

    dataset = dataset.map(_parse_function)

    window_datas = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    adjacencies = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    labels = tf.TensorArray(dtype=tf.uint8, size=0, dynamic_size=True)
    time_preread_indices = tf.TensorArray(dtype=tf.uint8, size=0, dynamic_size=True)
    window_indices = tf.TensorArray(dtype=tf.uint8, size=0, dynamic_size=True)

    for window_data, adjacency, label, time_preread_index, window_index in dataset:
        window_datas = window_datas.write(window_datas.size(), window_data)
        adjacencies = adjacencies.write(adjacencies.size(), adjacency)
        labels = labels.write(labels.size(), label)
        time_preread_indices = time_preread_indices.write(time_preread_indices.size(), time_preread_index)
        window_indices = window_indices.write(window_indices.size(), window_index)

    window_datas = window_datas.stack()
    adjacencies = adjacencies.stack()
    labels = labels.stack()
    time_preread_indices = time_preread_indices.stack()
    window_indices = window_indices.stack()

    return window_datas, adjacencies, labels, time_preread_indices, window_indices

def load_tfrecord_data_adjacency_label(tfrecord_path: str) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Load the TFRecord file and return window data, adjacency matrix, and labels as Tensors.

    Parameters:
    tfrecord_path (str): Path to the TFRecord file.

    Returns:
    Tuple[tf.Tensor, tf.Tensor, tf.Tensor]: A tuple containing:
        - window_datas (tf.Tensor): Tensor containing window data.
        - adjacencies (tf.Tensor): Tensor containing adjacency matrices.
        - labels (tf.Tensor): Tensor containing labels.
    """
    dataset = tf.data.TFRecordDataset(tfrecord_path)

    dataset = dataset.map(_parse_function)

    window_datas = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    adjacencies = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    labels = tf.TensorArray(dtype=tf.uint8, size=0, dynamic_size=True)

    for window_data, adjacency, label, *unused in dataset:
        window_datas = window_datas.write(window_datas.size(), window_data)
        adjacencies = adjacencies.write(adjacencies.size(), adjacency)
        labels = labels.write(labels.size(), label)

    window_datas = window_datas.stack()
    adjacencies = adjacencies.stack()
    labels = labels.stack()

    return window_datas, adjacencies, labels

# ------------------------------------- #
#  Start--Database_create--main func    #
# ------------------------------------- #

def database_create():
    """
    Process the data and create the corresponding TFRecord files for training, testing, and validation datasets.

    Functionality:
    This function reads sEMG data from CSV files for each gesture, processes it, and saves it as TFRecord files for each dataset type (train, test, val).
    """
    print("Processing the data, please wait...")
    print(
        f"Using the {cf.train_nums}-th data collection as the training set,\n"
        f"Using the {cf.test_nums}-th data collection as the test set,\n"
        f"Using the {cf.val_nums}-th data collection as the validation set.\n"
    )

    for gesture_number in cf.gesture:
        path = cf.data_path + f'original_data/sEMG_data{gesture_number}.csv'
        df = pd.read_csv(path, header=None).to_numpy()
        for dataset_type in ['train', 'val', 'test']:
            tfrecord_establish(df, gesture_number, dataset_type)
        print(f"Gesture {gesture_number} data processing completed.")

    print("data creation over")

# ------------------------------------- #
#  Over--Database_create--main func     #
# ------------------------------------- #
