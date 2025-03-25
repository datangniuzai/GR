#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/3/25 15:43
# @Author : Jason.LI
# @File : data_reading_base.py
# @Software: PyCharm

import os
import socket

import numpy as np


def data_save(data_save_path: str, reallocated_once_time:int = 4, collector_number: int = 8081, sample_rate:int = 2000):

    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_socket.bind(('192.168.1.100', collector_number))

    file_path = os.path.join(data_save_path, 'data.csv')
    print(f"Data will be saved at {file_path}")

    reallocated_data_size = reallocated_once_time * sample_rate
    output_data = np.zeros((reallocated_data_size, 64), dtype=np.float32)

    try:
        while 1:
            collected_samples = 0
            while collected_samples < reallocated_data_size:
                data, addr = udp_socket.recvfrom(1300)
                transposed_data = np.frombuffer(data[18:1298], dtype="<i2").reshape(10, 64) * 0.195
                output_data[collected_samples:collected_samples + 10, :] = transposed_data
                collected_samples += 10

            # Save data to file
            with open(file_path, 'a') as f:
                np.savetxt(f, output_data[:, :], delimiter=',', fmt='%.6f')

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        udp_socket.close()
        print("UDP socket closed. Program finished.")

