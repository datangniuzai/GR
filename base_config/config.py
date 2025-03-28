#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/3/25 19:57
# @Author : Jason.LI
# @File : base_config.py
# @Software: PyCharm

import ast
import datetime
import importlib.util
import logging
import os
import sys
import time
from pathlib import Path
from typing import List

from base_config.init_function import select_operation_mode, find_project_root, create_log_file, split_dataset_mode, split_data


class GlobalConfig:
    """Specified parameter"""

    # Channel number
    num_channels: int = 64
    # Sampling rate
    sample_rate: int = 2000
    # Collector number
    collector_number: int = 8081
    # Data folder path when use data
    path_to_use_data: str = "data/240908-LGJ-Man-S-17"
    # Input feature shape
    feature_shape: List = [6, 64, 5]

    # # Rest duration(second) between actions
    # action_rest:int = 20
    # Rest duration(second) between gestures
    gesture_rest: int = 12
    # Rest duration(second) between one loop
    loop_rest: int = 20
    # Gesture label list, can override the dataset's original parameters
    gesture_sequence = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    # Total read times for one gesture
    times_read_gesture: int = 20
    # Duration of each read
    once_read_time: int = 6
    # batch size
    batch_size: int = 32

    """ Dataset split parameter: Random"""
    # Training set count
    train_num: int = 12
    # Test set count
    test_num: int = 4
    # Validation set count
    val_num: int = 4
    """ Dataset split parameter: Specifying"""
    # Training set indices
    train_indices: List = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    # Test set indices
    test_indices: List = [16]
    # Validation set indices
    val_indices: List = [17, 18, 19, 20]

    # Window size
    window_size = 400
    # Window step size
    step_size = 200
    # Small window size
    window_size_little = 150
    # Small window step size
    step_size_little = 50
    # Number of epochs
    epochs = 3
    # Model path
    model_path: str = None
    """Input parameters"""
    # Mode
    # Data Reading and Saving;
    # Model Training;
    # Data Analysis;
    # Online Display;
    mode_use: str = None
    # Dataset selection mode: random /Specified
    tvt_select_mode: str = None

    """Generated parameters"""

    # project root path
    project_root: Path = None
    # log path
    log_path: Path = None

    # Unused data
    remaining_numbers: list = None
    # data path to use when saving data
    path_to_save_data: str = None

    # Directory where all outputs from training (models, figures etc.) will be stored
    train_folder_path: str = None
    # File path to save training metrics history (e.g., loss, accuracy) in CSV format and training_config.txt.
    train_config_txt_path: str = None
    # File path to save test results (e.g. acc, recall)
    test_results_csv_path: str = None
    # File path to save training history (e.g. loss, accuracy)
    history_csv_file:str = None

    # Number of gestures
    gesture_num: int = None

    # Model
    model = None
    # Model name
    model_name: str = None

    # Model training start time
    start_train_time = None
    # Model training end time
    end_train_time = None
    # Model training time
    train_duration_minutes = None

    @classmethod
    def update_param(cls, param_name, value):
        """
        Dynamically update a parameter's value.
        """
        if hasattr(cls, param_name):
            setattr(cls, param_name, value)
            logging.info(f"Updated parameter '{param_name}' to '{value}'")
            print(f"Updated parameter '{param_name}' to '{value}'")
        else:
            print(f"⚠️ Parameter '{param_name}' not found in Config.")

    @classmethod
    def display_config(cls):
        """
        Display configuration parameters in categorized sections,
        strictly following the original comment grouping.
        """
        # Get non-method attributes
        config_items = {k: v for k, v in cls.__dict__.items() if not k.startswith("__") and not callable(v)}

        if not config_items:
            print("No configuration parameters found")
            return

        # Calculate max key length for alignment
        max_len = max(len(str(k)) for k in config_items.keys()) + 2

        # Section titles in EXACT original order
        sections = ["Specified parameter", "Dataset split parameters", "Input parameters", "Generated parameters"]

        # Track displayed params to find leftovers
        displayed_params = set()

        # Print each section with original spacing
        for section in sections:
            print(f'\n""" {section} """')
            for param, value in config_items.items():
                # Find params that belong to this section by code position
                if (
                    (
                        section == "Specified parameter"
                        and param
                        in [
                            "num_channels",
                            "sample_rate",
                            "collector_number",
                            "path_to_use_data",
                            "feature_shape",
                            "gesture_rest",
                            "loop_rest",
                            "gesture_sequence",
                            "times_read_gesture",
                            "once_read_time",
                            "window_size",
                            "step_size",
                            "window_size_little",
                            "step_size_little",
                            "epochs",
                            "model_path"
                            "batch_size"
                        ]
                    )
                    or (
                        section == "Dataset split parameters"
                        and param
                        in ["train_num", "test_num", "val_num", "train_indices", "test_indices", "val_indices"]
                    )
                    or (section == "Input parameters" and param in ["mode_use", "tvt_select_mode"])
                    or (
                        section == "Generated parameters"
                        and param
                        in [
                            "project_root",
                            "log_path",
                            "remaining_numbers",
                            "gesture_num",
                            "path_to_save_data",
                            "train_folder_path",
                            "train_config_txt_path",
                            "history_csv_file"
                            "test_results_csv_path",
                            "model",
                            "model_name",
                            "start_train_time",
                            "end_train_time",
                            "train_duration_minutes"
                        ]
                    )
                ):
                    print(f"  {param.ljust(max_len)} : {value}")
                    displayed_params.add(param)

        # Print any remaining parameters (shouldn't happen if sections are complete)
        # remaining = {k: v for k, v in config_items.items() if k not in displayed_params}
        # if remaining:
        #     print('\n""" Unclassified Parameters """')
        #     for param, value in remaining.items():
        #         print(f"  {param.ljust(max_len)} : {value}")

    @classmethod
    def config_init(cls, log_path: str = None):
        """
        Initialize all parameters.
        mode_dict = {'data_reading_and_saving','model_training','data_analysis','online_display'}
        """

        cls.project_root = find_project_root()

        cls.mode_use = select_operation_mode()

        if log_path is None:
            log_path = f"logs/{cls.mode_use}_{datetime.datetime.now().strftime('%m-%d_%H-%M')}.log"
        cls.log_path = create_log_file(root_path=cls.project_root, log_path=log_path)

        logging.info(f"The usage mode this time is: {cls.mode_use}")

        logging.info(f"Changing working directory to: {cls.project_root}")
        os.chdir(cls.project_root)
        logging.info(f"Successfully changed working directory to: {os.getcwd()}")

        time.sleep(0.01)

        if cls.mode_use in ["model_training", "data_analysis"]:

            if cls.path_to_use_data[-1] != "/":
                cls.path_to_use_data += "/"

            # 1. Get dataset split mode
            data_set_mode, cls.tvt_select_mode = split_dataset_mode()

            # 2. Load and update configuration
            if data_set_mode == "1" or data_set_mode == "2":
                cls.update_global_config()

            # 3. Handle random split mode
            if data_set_mode == "1":
                # Validate sample counts
                requested_samples = cls.train_num + cls.val_num + cls.test_num
                if requested_samples > cls.times_read_gesture:
                    raise ValueError(
                        f"Dataset configuration error: "
                        f"Requested {requested_samples} samples (train+val+test) "
                        f"exceeds available {cls.times_read_gesture} collected samples. "
                        f"Please adjust your dataset split ratios."
                    )

                # Split data if validation passes
                (cls.train_indices, cls.test_indices, cls.val_indices, cls.remaining_numbers) = split_data(
                    cls.times_read_gesture, cls.train_num, cls.test_num, cls.val_num
                )

            elif data_set_mode == "2":
                # Convert to sets for intersection checking
                train_set = set(cls.train_indices)
                test_set = set(cls.test_indices)
                val_set = set(cls.val_indices)

                # Critical validation 1: ensure mutually exclusive splits
                if train_set & test_set or train_set & val_set or test_set & val_set:
                    raise ValueError(
                        "DATA LEAKAGE ALERT: Detected overlapping samples in train/val/test splits. "
                        "All splits must contain completely distinct indices."
                    )

                # Critical validation 2: check all indices are within valid range
                all_indices = train_set.union(test_set).union(val_set)
                max_index = max(all_indices) if all_indices else 0
                if max_index > cls.times_read_gesture:
                    raise ValueError(
                        f"Index out of bounds: Found index {max_index} "
                        f"which exceeds maximum available samples ({cls.times_read_gesture}). "
                        f"Please check your specified indices."
                    )

                # Update counts and validate total samples
                cls.train_num = len(cls.train_indices)
                cls.test_num = len(cls.test_indices)
                cls.val_num = len(cls.val_indices)
                requested_samples = cls.train_num + cls.val_num + cls.test_num

                if requested_samples > cls.times_read_gesture:
                    raise ValueError(
                        f"Dataset configuration error: "
                        f"Requested {requested_samples} samples (train+val+test) "
                        f"exceeds available {cls.times_read_gesture} collected samples. "
                        f"Please adjust your dataset split ratios."
                    )

            elif data_set_mode == "3":
                # Load parameters from existing dataset info file

                info_file_path = os.path.join(cls.path_to_use_data, "processed_data", "tf_data_info.txt")

                try:
                    with open(info_file_path, "r") as f:
                        lines = f.readlines()

                    param_dict = {}
                    for line in lines:
                        if ":" in line and not line.startswith("Data Processing") and not line.startswith("Generated"):
                            key, value = line.split(":", 1)
                            param_dict[key.strip()] = ast.literal_eval(value.strip())

                    # Update class attributes
                    cls.train_indices = param_dict["train_indices"]
                    cls.test_indices = param_dict["test_indices"]
                    cls.val_indices = param_dict["val_indices"]
                    cls.remaining_numbers = param_dict["remaining_numbers"]

                    cls.train_num = len(cls.train_indices)
                    cls.test_num = len(cls.test_indices)
                    cls.val_num = len(cls.val_indices)

                    cls.window_size = param_dict["window_size"]
                    cls.step_size = param_dict["step_size"]
                    cls.window_size_little = param_dict["window_size_little"]
                    cls.step_size_little = param_dict["step_size_little"]
                    cls.gesture_sequence = param_dict["gesture_sequence"]
                    # Calculate feature shape

                    logging.info(f"Successfully loaded dataset parameters from {info_file_path}")

                except FileNotFoundError:
                    raise FileNotFoundError(
                        f"Existing dataset info file not found at {info_file_path}. "
                        "Please ensure you have processed data first or choose another mode."
                    )
                except Exception as e:
                    raise ValueError(
                        f"Failed to parse dataset info file: {str(e)}. "
                        "Please check the file format or reprocess your data."
                    )

            cls.gesture_num =  len(cls.gesture_sequence)
            cls.feature_shape = [(cls.window_size - cls.window_size_little) // cls.step_size_little + 1, 64, 5]

    @classmethod
    def update_global_config(cls):
        """Load and validate DataConfig against GlobalConfig parameters"""

        def load_data_config(path_to_use_data):
            config_path = Path(path_to_use_data) / "data_config.py"
            spec = importlib.util.spec_from_file_location("data_config", str(config_path))
            data_config_path = importlib.util.module_from_spec(spec)
            sys.modules["data_config"] = data_config_path
            spec.loader.exec_module(data_config_path)
            return data_config_path.DataConfig

        data_config_class = load_data_config(cls.path_to_use_data)
        data_config = data_config_class()
        data_config.display_config()

        # Validate gesture sequence
        if hasattr(data_config, "gesture_sequence"):
            config_numbers = set(cls.gesture_sequence)
            data_numbers = set(data_config.gesture_sequence)

            # Check for extra numbers in GlobalConfig
            extra_numbers = config_numbers - data_numbers
            if extra_numbers:
                logging.error(
                    f"Gesture sequence validation FAILED\n"
                    f"GlobalConfig contains extra numbers: {sorted(extra_numbers)}\n"
                    f"GlobalConfig sequence: {cls.gesture_sequence}\n"
                    f"DataConfig sequence: {data_config.gesture_sequence}"
                )
                sys.exit(1)

            # Handle sequence order mismatch
            if data_config.gesture_sequence != cls.gesture_sequence:
                cls._prompt_param_choice(
                    param_name="gesture_sequence",
                    config_value=cls.gesture_sequence,
                    data_config_value=data_config.gesture_sequence,
                )

        # Validate times_read_gesture
        if hasattr(data_config, "times_read_gesture"):
            if cls.times_read_gesture > data_config.times_read_gesture:
                logging.error(
                    f"Gesture repetition count mismatch\n"
                    f"GlobalConfig repetitions ({cls.times_read_gesture}) > "
                    f"DataConfig repetitions ({data_config.times_read_gesture})\n"
                    f"Terminating to prevent data collection issues."
                )
                sys.exit(1)
            elif cls.times_read_gesture != data_config.times_read_gesture:
                cls._prompt_param_choice(
                    param_name="times_read_gesture",
                    config_value=cls.times_read_gesture,
                    data_config_value=data_config.times_read_gesture,
                )

        # Auto-adjust these parameters without prompt
        cls.gesture_rest = data_config.gesture_rest
        cls.loop_rest = data_config.loop_rest
        cls.once_read_time = data_config.once_read_time

    @classmethod
    def _prompt_param_choice(cls, param_name, config_value, data_config_value):
        """Helper function to handle parameter choice prompts"""
        while True:
            time.sleep(0.01)
            choice = (
                input(
                    f"{param_name}: GlobalConfig ({config_value}) ≠ DataConfig ({data_config_value}). "
                    f"Use GlobalConfig? [Y/N]: "
                )
                .strip()
                .upper()
            )

            if choice == "Y":
                logging.info(f"Using GlobalConfig's {param_name}")
                return
            elif choice == "N":
                setattr(cls, param_name, data_config_value)
                logging.info(f"Using DataConfig's {param_name}")
                return
            else:
                print("Invalid input. Please enter Y or N.")


if __name__ == "__main__":

    cf = GlobalConfig()
    cf.config_init()
    cf.display_config()
