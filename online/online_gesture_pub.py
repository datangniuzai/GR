#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/3/6 11:34
# @Author : Jason.LI
# @File : online_gesture_pub.py
# @Software: PyCharm

import datetime
import logging
import os
import socket
import threading
from multiprocessing import Queue

import numpy as np
import tensorflow as tf
import zmq

import config as cf
from gesture_recognition.script.dataset import primary_window_feature
from gesture_recognition.script.filtering import bandpass_and_notch_filter
from model_file import litestfnet_model_creat


def original_data_receiver(data_queue: Queue, window_size: int, port: int = 8080) -> None:
    """
    Receives UDP packets, processes sEMG data, and stores it in a queue.

    Args:
        data_queue: Queue to store processed data windows (shape: [window_size, 64]).
        window_size: Size of the data window to collect before enqueueing.
        port: Local data receiving port.
    """
    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_socket.bind(("192.168.1.100", port))

    try:
        output_data = np.empty((window_size, 64), dtype=np.float32)
        idx = 0

        while True:
            data, addr = udp_socket.recvfrom(1300)
            # logging.info(f"Received data from {addr}")

            transposed_data = np.frombuffer(data[18:1298], dtype="<i2").reshape(10, 64) * 0.195
            output_data[idx : idx + 10, :] = transposed_data
            idx += 10

            if idx == window_size:

                # Save data to file
                # with open(cf.data_path + f'original_data/origin_data.csv', 'a') as f:
                #     np.savetxt(f, output_data, delimiter=',', fmt='%.6f')

                data_queue.put(output_data[:])
                idx = 0

    except Exception as e:
        logging.error(f"Error in receiver: {e}", exc_info=True)
        raise
    finally:
        udp_socket.close()
        logging.info("UDP socket closed")


def model_predictor(data_queue, path_model, window_size_little, step_size_little) -> None:
    """
    A function to predict gestures from data in a queue using a pre-trained model.

    Args:
        data_queue (Queue[np.ndarray]): A queue containing input data arrays for prediction.
        path_model (str): The file path to the pre-trained model weights.
        window_size_little: the size of secondary window.
        step_size_little: the step of secondary window.
    """
    context = zmq.Context()
    zmq_socket = context.socket(zmq.PUB)
    zmq_socket.bind("tcp://*:5555")

    model = litestfnet_model_creat()
    model.load_weights(path_model)
    logging.info("Model loaded successfully")
    logging.info("Prediction start")
    while True:
        try:
            if not data_queue.empty():
                original_data = data_queue.get()

                filtered_data = bandpass_and_notch_filter(original_data)
                window_data_feature = tf.expand_dims(
                    tf.convert_to_tensor(
                        primary_window_feature(filtered_data, window_size_little, step_size_little),
                        dtype=tf.float32,
                    ),
                    axis=0,
                )
                predicted_class_index = np.argmax(model.predict(window_data_feature), axis=1)
                gesture = int(predicted_class_index[0]) + 1

                zmq_socket.send_string(str(gesture))

                logging.info(f"Predicted gesture: {gesture}")

        except Exception as e:
            logging.error(f"Error in predictor: {e}", exc_info=True)
            continue


if __name__ == "__main__":
    cf.config_read()

    log_file_path = f"logs/online_show_{datetime.datetime.now().strftime('%m-%d_%H-%M')}.log"

    if not os.path.exists(os.path.dirname(log_file_path)):
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file_path), logging.StreamHandler()],
    )

    sEMG_data_queue = Queue()
    # 🔥change the model path here！🔥
    model_path = "data/online_data_test/all_train_info/LiteSTFNet_03-22_20-29/models/model_30.keras"


    receiver_thread = threading.Thread(
        target=original_data_receiver, args=(sEMG_data_queue, cf.window_size)
    )

    predictor_thread = threading.Thread(
        target=model_predictor, args=(sEMG_data_queue, model_path, cf.window_size_little, cf.step_size_little)
    )

    receiver_thread.start()
    predictor_thread.start()

    receiver_thread.join()
    predictor_thread.join()
