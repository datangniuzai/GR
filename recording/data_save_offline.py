#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/3/24 19:58
# @Author : Jason.LI
# @File : data_save_offline.py
# @Software: PyCharm

import os
import time
import socket
import logging
import datetime
from pathlib import Path
from typing import List, Union

import pyttsx3
import numpy as np

from base_config.config import GlobalConfig


def create_data_folder(base_root_path: Union[str, Path]) -> str:
    """Create timestamped data directory structure under specified base path.

    Directory Structure:
    [base_root_path]/
    └── data/
        └── YYYY-MM-DD_HH-MM-SS/
            ├── processed_data/    # Cleaned/normalized datasets
            ├── original_data/     # Raw collected data (immutable)
            ├── picture/           # Visualization outputs (plots/charts)
            └── all_train_info/    # Training metadata and logs

    Args:
        base_root_path: Parent directory where 'data' folder will be created

    Returns:
        str: Relative path from base_root_path to created folder
             (format: "data/YYYY-MM-DD_HH-MM-SS/")
    """
    # Convert Path to string if necessary
    if isinstance(base_root_path, Path):
        data_root = os.path.join(str(base_root_path), "data")
    else:
        data_root = os.path.join(base_root_path, "data")
    os.makedirs(data_root, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    timestamp_dir = os.path.join(data_root, timestamp)

    # Core directory structure
    sub_dirs = (
        "processed_data",  # For processed/cleaned data files
        "original_data",  # For raw unprocessed data
        "picture",  # For visualization outputs
        "all_train_info",  # For training logs and metadata
    )

    # Create all directories
    for dir_name in sub_dirs:
        os.makedirs(os.path.join(timestamp_dir, dir_name), exist_ok=True)

    abs_path = os.path.abspath(timestamp_dir)

    print(f"[System] Experiment directory initialized at:\n{abs_path}")
    logging.info(f"[System] Experiment directory initialized at:\n{abs_path}")

    return f"data/{timestamp}/"


def write_config_file(
    gesture_sequence: List,
    times_read_gesture: int,
    once_read_time: int,
    gesture_rest: int,
    loop_rest: int,
    data_folder_path: str,
    start_time: str,
    end_time: str,
):
    content = f'''\
class DataConfig:
    """
    Configuration class for managing hand gesture experiment parameters.

    ⚠️ Volunteer and Data Information:
    - Name: []
    - Age: [] years
    - Gender:  [] /e.g. Male / Female
    - Measured Arm: [] /e.g. Left / Right
    - Gesture Rest Duration: [] seconds between gestures
    - Loop Rest Duration: [] seconds between each full loop of gestures
    - Diet: [] /e.g.Normal
    - Weekly Exercise: [] hours
    - Neurological Diseases: [] /e.g. None
    - Physical Conditions: [] /e.g. None
    - Sleep (Before Experiment): [] hours (Bedtime: /e.g.2024-11-17 23:00)

    Data Collection Period:
        - Data Collection Start Time: {start_time}
        - Data Collection End Time: {end_time}

    Experiment Details:
    - Identifier Format: YYYYMMDD-Name-Gender-StaticOrDynamic-GestureCount
    - Example Identifier: "240909-LJX-Man-S-17"
    """

    def __init__(self):
        self.gesture_sequence = {gesture_sequence}
        self.times_read_gesture = {times_read_gesture}
        self.once_read_time = {once_read_time}
        self.gesture_rest = {gesture_rest}
        self.loop_rest = {loop_rest}

    def display_config(self):
        """
        Displays the current configuration parameters for the hand gesture experiment.
        """
        print("=== Experiment Configuration ===")
        print(f"Gesture Sequence: {{self.gesture_sequence}}")
        print(f"Times to Read Each Gesture: {{self.times_read_gesture}}")
        print(f"Read Duration per Action (seconds): {{self.once_read_time}}")
        print(f"Gesture Rest Duration (seconds): {{self.gesture_rest}}")
        print(f"Loop Rest Duration (seconds): {{self.loop_rest}}")
        print("================================")
'''
    file_path = os.path.join(data_folder_path, "data_config.py")
    with open(file_path, "w", encoding="utf-8") as f:  # 明确指定utf-8编码
        f.write(content)

    print(f"✅ Data configuration file '{file_path}' has been generated successfully!")
    logging.info(f"✅ Data configuration file '{file_path}' has been generated successfully!")


def sEMG_data_save_offline(
    collector_number: int,
    once_read_time: int,
    sample_rate: int,
    times_read_gesture: int,
    gesture_sequence: List,
    path_to_save_data: str,
    gesture_rest: int,
    loop_rest: int,
):

    engine = pyttsx3.init()
    rate = engine.getProperty("rate")
    engine.setProperty("rate", rate + 50)  # Increase speech rate
    voices = engine.getProperty("voices")
    engine.setProperty("voice", voices[1].id)

    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_socket.bind(("192.168.1.100", collector_number))

    reallocated_data_size = (once_read_time + 1) * sample_rate
    output_data = np.zeros((reallocated_data_size, 64), dtype=np.float32)

    start_time = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    try:
        i = 1
        while i < (times_read_gesture + 1):
            for gesture_number in gesture_sequence:
                text_to_speak = f"Please prepare for gesture {gesture_number}, collection starting."
                print(text_to_speak)
                engine.say(text_to_speak)
                engine.runAndWait()
                time.sleep(0.5)
                print("Collecting data...")

                collected_samples = 0
                while collected_samples < reallocated_data_size:
                    data, addr = udp_socket.recvfrom(1300)
                    transposed_data = np.frombuffer(data[18:1298], dtype="<i2").reshape(10, 64) * 0.195
                    output_data[collected_samples : collected_samples + 10, :] = transposed_data
                    collected_samples += 10

                # Save data to file
                with open(path_to_save_data + f"original_data/sEMG_data{gesture_number}.csv", "a") as f:
                    np.savetxt(f, output_data[sample_rate:, :], delimiter=",", fmt="%.6f")
                time.sleep(0.5)
                text_to_speak = "Please rest."
                print(text_to_speak)
                engine.say(text_to_speak)
                engine.runAndWait()
                time.sleep(gesture_rest)

            i += 1
            time.sleep(loop_rest)

    finally:
        end_time = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

        write_config_file(
            gesture_sequence,
            times_read_gesture,
            once_read_time,
            gesture_rest,
            loop_rest,
            cf.path_to_save_data,
            start_time,
            end_time,
        )

        udp_socket.close()

        print(
            f"\u2764 Please rename the folder [{path_to_save_data}] to identifier "
            "and complete the details of the comment."
        )


if __name__ == "__main__":

    # init global config
    cf = GlobalConfig()
    cf.config_init()
    cf.display_config()

    cf.update_param("path_to_save_data", create_data_folder(str(cf.project_root)))

    sEMG_data_save_offline(
        cf.collector_number,
        cf.once_read_time,
        cf.sample_rate,
        cf.times_read_gesture,
        cf.gesture_sequence,
        cf.path_to_save_data,
        cf.gesture_rest,
        cf.loop_rest,
    )
