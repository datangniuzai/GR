#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2024/11/14 19:43
# @Author : Jiaxuan LI
# @File : filtering.py
# @Software: PyCharm

import os
import json
import time
import socket
import struct
import pyttsx3
import datetime
import numpy as np

import config as cf


def sEMG_data_read_save():
    """
    Collect and save sEMG data by receiving UDP packets.

    Instructions:
    1. Configure parameters in the `cf` module (collector_number, gesture, etc.).
    2. Run the function to start data collection.
    3. Ensure the UDP socket is properly configured with the correct IP and port.

    Returns:
    None
    """
    engine = pyttsx3.init()
    rate = engine.getProperty('rate')
    engine.setProperty('rate', rate + 50)  # Increase speech rate
    # voices = engine.getProperty('voices')
    # engine.setProperty('voice', voices[1].id)

    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_socket.bind(('192.168.1.100', cf.collector_number))

    reallocated_data_size = (cf.time_preread + 1) * cf.sample_rate
    output_data = np.zeros((reallocated_data_size, 64), dtype=np.float32)

    start_time = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    try:
        i = 1
        while i < (cf.turn_read_sum + 1):
            for gesture_number in cf.gesture:

                text_to_speak = f"Please prepare for gesture {gesture_number}, collection starting."
                print(text_to_speak)
                engine.say(text_to_speak)
                engine.runAndWait()
                time.sleep(0.5)
                print("Collecting data...")

                collected_samples = 0
                while collected_samples < reallocated_data_size:
                    data, addr = udp_socket.recvfrom(1300)
                    reshaped_data = np.frombuffer(data[18:1298], dtype='<i2').reshape(10, 64)
                    output_data[collected_samples:collected_samples + 10, :] = reshaped_data
                    collected_samples += 10

                # Save data to file
                with open(cf.data_path + f'original_data/sEMG_data{gesture_number}.csv', 'a') as f:
                    np.savetxt(f, output_data[cf.sample_rate:, :] * 0.195, delimiter=',', fmt='%.6f')

                text_to_speak = "Please rest."
                print(text_to_speak)
                engine.say(text_to_speak)
                engine.runAndWait()
                time.sleep(15)

            i += 1
            time.sleep(180)

    finally:
        end_time = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

        generate_volunteer_experiment_info(start_time, end_time)

        udp_socket.close()

        print(f"\u2764Please rename the folder [{cf.data_path}] to identifier "
              "and complete the details of the [vol_exp_info.json].")

def generate_volunteer_experiment_info(start_time,end_time):

    experiment_info = {
        "name": "volunteer_experiment_info",
        "description": "Details of subjects and experimental process",
        "detailed description": "",
        "note": "The following description of time is in hours.",
        "explanation about identifier": "date/subject's_last_name/man_or_female/static_or_dynamic/number_of_gestures",
        "identifier": "250112-Z-Man-S-26",
        "volunteer_info": {
            "name": "",
            "age": "",
            "gender": "male/female",
            "measured_arm": "left/right",
            "diet": "Normal",
            "weekly_exercise_duration": 3.5,
            "subject_conditions": {
                "neurological_diseases": "None",
                "physical_conditions": "Healthy",
                "sleep": {
                    "previous_night_sleep_duration": 7.5,
                    "bedtime": "2024-11-17T23:00:00"
                },
                "diet": "Normal",
                "weekly_exercise_duration": 3.5
            }
        },
        "experiment_info": {
            "gesture_sequence": cf.gesture,
            "collector_number": cf.collector_number - 8079,
            "gesture_read_count_per_instance":cf.turn_read_sum,
            "read_duration_per_instance": cf.time_preread,
            "experiment_time": {
                "start_time": start_time,
                "end_time": end_time,
                "gesture_rest": cf.gesture_rest,
                "action_rest":cf.action_rest
            }
        }
    }
    file_name = os.path.join(cf.data_path, "vol_exp_info.json")
    with open(file_name, 'w', encoding='utf-8') as f:
        json.dump(experiment_info, f, ensure_ascii=False, indent=4)

    print(f"实验记录数据已保存至: {file_name}")


