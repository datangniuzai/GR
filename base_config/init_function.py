#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/3/25 19:37
# @Author : Jason.LI
# @File : init_function.py
# @Software: PyCharm

import os
import sys
import time
import random
import logging
import datetime
from pathlib import Path
from typing import List, Tuple,Union


def select_operation_mode() -> str:
    """
    Prompts the user to select the mode using a dictionary and returns the selected mode.

    Returns:
        str: A string representing the selected mode ('Data Reading and Saving',
             'Model Training', 'Data Analysis', or 'Online Display').
    """
    mode_dict = {"1": "data_reading_and_saving", "2": "model_training", "3": "data_analysis", "4": "online_display"}

    while True:
        print("================================")
        pattern_mode = input(
            "Please select the mode:\n"
            "1. Data Reading and Saving;\n"
            "2. Model Training;\n"
            "3. Data Analysis;\n"
            "4. Online Display;\n"
            "(Enter your choice): "
        )
        print("================================")
        if pattern_mode in mode_dict:
            return mode_dict[pattern_mode]
        else:
            print("⚠️ Invalid input! Please enter '1', '2', '3', or '4' to select the mode.\n")

def split_dataset_mode() -> tuple[str, str]:
    """
    Prompts the user to choose the dataset split mode and returns the selected mode and its description.

    Returns:
        tuple: A tuple containing:
            - mode choice ('1', '2', or '3')
            - string description ('Random', 'Specified', or 'Existing')
    """
    while True:
        time.sleep(0.01)
        data_set_mode = input(
            "Please choose the dataset split mode:\n"
            "1. Random split\n"
            "2. Specified split\n"
            "3. Use existing dataset\n"
            "Enter your choice (1/2/3): "
        ).strip()

        if data_set_mode in ["1", "2", "3"]:
            mode_description = {
                "1": "Random",
                "2": "Specified",
                "3": "Existing"
            }[data_set_mode]
            logging.info(f"User selected: {mode_description} split mode")
            print(f"User selected: {mode_description} split mode")
            time.sleep(0.01)
            print("================================")
            return data_set_mode, mode_description

        print(f"⚠️ Invalid input: '{data_set_mode}'. Please enter 1, 2, or 3.\n")



def find_project_root(start_path: Path = None) -> Path:
    """
    Find the project root directory by searching for the 'README.md' file upwards from the starting path.

    Args:
        start_path (Path, optional): The directory from which to start the search. If None, it defaults
                                      to the current script's directory.

    Returns:
        Path: The path of the project root directory where 'README.md' is found.

    Raises:
        FileNotFoundError: If no 'README.md' is found while traversing up the directories.
    """
    if start_path is None:
        start_path = Path(__file__).resolve()

    while start_path != start_path.parent:
        if (start_path / "README.md").exists():
            return start_path
        start_path = start_path.parent

    raise FileNotFoundError(
        "Project root directory not found. "
        "Please ensure the current directory contains a README.md file, "
        "or modify the root directory search criteria as needed."
    )


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

def create_log_file(
        root_path: Union[str, Path],
        log_path: str,
        level: str = "INFO",
        filemode: str = "a",
        console_output: bool = False,
        enable_unicode: bool = True
) -> Path:
    """
    Create a .log file with configurable logging output and Unicode support.

    Args:
        root_path: Directory where the log file will be created (str or Path)
        log_path: Name of the log file
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        filemode: File writing mode ('w' for overwrite, 'a' for append)
        console_output: Whether to print logs to terminal
        enable_unicode: Whether to force UTF-8 encoding for Unicode symbols

    Returns:
        Path to the created log file

    Raises:
        ValueError: If invalid logging level is provided
    """
    # Convert and validate paths
    root_path = Path(root_path)
    os.makedirs(root_path, exist_ok=True)
    log_file = root_path / log_path

    # Validate logging level
    log_level = getattr(logging, level.upper(), None)
    if not isinstance(log_level, int):
        raise ValueError(f"Invalid logging level: {level}")

    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # Clear existing handlers
    logger.handlers.clear()

    # File handler with UTF-8 encoding
    file_handler = logging.FileHandler(
        log_file,
        mode=filemode,
        encoding='utf-8'  # Force UTF-8 encoding
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(file_handler)

    # Console handler with Unicode support
    if console_output:
        stream_handler = logging.StreamHandler(
            stream=sys.stdout if enable_unicode else None
        )
        stream_handler.setLevel(log_level)
        stream_handler.setFormatter(
            logging.Formatter("%(levelname)s - %(message)s")
        )
        logger.addHandler(stream_handler)

    # Initial log message (Unicode supported)
    init_msg = f"📝 Logging initialized (file: {log_file}, console: {console_output}, Unicode: {enable_unicode})"
    logger.info(init_msg)

    if not console_output:
        print(init_msg)

    return log_file

def split_data(_turn_read_sum: int, _train_num: int, _test_num: int, _val_num: int) \
        -> Tuple[List[int], List[int], List[int], List[int]]:
    """
    Split the dataset into training, testing, validation sets and return any remaining numbers.

    Parameters:
    - turn_read_sum: The total number of samples in the dataset (int).
    - train_num: The number of samples in the training set (int).
    - test_num: The number of samples in the testing set (int).
    - val_num: The number of samples in the validation set (int).

    Returns:
    - A tuple of four lists:
      - train_nums: A list of sample indices for the training set (List[int]).
      - test_nums: A list of sample indices for the testing set (List[int]).
      - val_nums: A list of sample indices for the validation set (List[int]).
      - remaining_nums: A list of remaining sample indices (List[int]).
    """
    all_numbers = list(range(1, _turn_read_sum + 1))  # List of all sample indices

    _train_nums = random.sample(all_numbers, _train_num)

    _remaining_numbers = [num for num in all_numbers if num not in _train_nums]
    _test_nums = random.sample(_remaining_numbers, _test_num)


    _remaining_numbers = [num for num in _remaining_numbers if num not in _test_nums]
    _val_nums = random.sample(_remaining_numbers, _val_num)
    _remaining_numbers = [num for num in _remaining_numbers if num not in _val_nums]

    return _train_nums, _test_nums, _val_nums, _remaining_numbers
