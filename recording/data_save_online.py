#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/3/24 19:58
# @Author : Jason.LI
# @File : data_save_online.py
# @Software: PyCharm

import datetime
import logging
import logging.handlers
import multiprocessing
import os
import socket
import time
from queue import Queue
from typing import List

import numpy as np
import pyttsx3

import config as cf


def setup_logger(log_queue: Queue):
    handler = logging.handlers.QueueHandler(log_queue)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)


def log_listener(log_queue: Queue,data_path:str):

    log_file_path = data_path + f"logs/online_data_{datetime.datetime.now().strftime('%m-%d_%H-%M')}.log"
    if not os.path.exists(os.path.dirname(log_file_path)):
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    listener = logging.getLogger()
    listener.addHandler(file_handler)

    while True:
        try:
            if not log_queue.empty():
                record = log_queue.get()
                listener.handle(record)
            else:
                time.sleep(0.1)
        except Exception as e:
            logging.error(f"Error in log_listener: {e}")
            break

def sEMG_data_save_online(window_size: int, data_queue: Queue, log_queue: Queue, collector_number: int = 8081):

    setup_logger(log_queue)
    logging.info("Started EMG data collection")

    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_socket.bind(('192.168.1.100', collector_number))

    try:
        output_data = np.empty((window_size, 64), dtype=np.float32)
        idx = 0

        while True:
            data, addr = udp_socket.recvfrom(1300)
            transposed_data = np.frombuffer(data[18:1298], dtype="<i2").reshape(10, 64) * 0.195
            output_data[idx: idx + 10, :] = transposed_data
            idx += 10

            if idx == window_size:
                data_queue.put(output_data[:])
                idx = 0

    except Exception as e:
        logging.error(f"Error in receiver: {e}", exc_info=True)
        raise
    finally:
        udp_socket.close()
        logging.info("UDP socket closed")

def save_data(data_queue:Queue, data_path:str, log_queue: Queue):

    setup_logger(log_queue)
    logging.info("The data saving process starts.")

    data_save_path  = data_path + f"original_data/online_data.csv"
    logging.info(f"Data saved at {data_save_path}.")

    while True:
        if not data_queue.empty():
            original_data = data_queue.get()
            with open(data_save_path, 'a') as f:
                np.savetxt(f, original_data, delimiter=',', fmt='%.6f')
            logging.info("Data saved successfully Once.")

def voice_prompt(gesture_list: List[str], gesture_duration: int, rest_duration: int, log_queue: Queue):
    def speak_text(text_to_speak: str):
        print(text_to_speak)
        engine.say(text_to_speak)
        engine.runAndWait()

    setup_logger(log_queue)
    logging.info("Started voice prompts process.")

    engine = pyttsx3.init()
    rate = engine.getProperty('rate')
    engine.setProperty('rate', rate + 50)

    speak_text("Please prepare.")

    logging.info(f"Starting!")

    time.sleep(5)

    for i, gesture in enumerate(gesture_list):
        speak_text(f"Gesture {gesture} for {gesture_duration} seconds.")
        logging.info(f"Gesture {gesture} recording start")
        time.sleep(gesture_duration)
        speak_text(f"Please rest for {rest_duration} seconds.")
        logging.info(f"Rest period start")
        time.sleep(rest_duration)
    speak_text("Data collection complete!")
    logging.info("Voice prompt process completed.")


def main():
    cf.config_read()
    log_queue = Queue()

    window_size = 500
    collector_port = 8081
    data_save_path = cf.data_path
    gesture_list = cf.gesture
    gesture_duration = 12
    rest_duration = 6

    listener_process = multiprocessing.Process(target=log_listener, args=(log_queue,data_save_path))
    listener_process.start()

    data_queue = Queue()
    emg_process = multiprocessing.Process(target=sEMG_data_save_online, args=(window_size, data_queue, log_queue, collector_port))
    save_process = multiprocessing.Process(target=save_data, args=(data_queue, data_save_path, log_queue))
    voice_process = multiprocessing.Process(target=voice_prompt, args=(gesture_list, gesture_duration, rest_duration, log_queue))

    emg_process.start()
    save_process.start()
    voice_process.start()

    try:
        while voice_process.is_alive():
            time.sleep(1)

        print("Voice prompt process has finished. Terminating all processes...")
        emg_process.terminate()
        save_process.terminate()
        emg_process.join()
        save_process.join()

        print("All processes terminated.")
    except KeyboardInterrupt:
        print("\nTerminating all processes...")
        emg_process.terminate()
        save_process.terminate()
        voice_process.terminate()
        emg_process.join()
        save_process.join()
        voice_process.join()
        print("All processes terminated.")

    log_queue.put(None)
    listener_process.join()

if __name__ == "__main__":
    main()