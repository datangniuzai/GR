#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/1/28 13:40
# @Author : JIAXUAN LI
# @File : online_show.py
# @Software: PyCharm

import os
import sys
import socket

import numpy as np
import tensorflow as tf
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from multiprocessing import Queue, Process, Event
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QPushButton

import config as cf
from filtering import bandpass_and_notch_filter
from model_file import tccnn_model_creat
from dataset import calc_td


os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def online_data_read(data_queue: Queue, event: Event, window_size) -> None:
    """
    Reads UDP data, processes it, and puts it into a queue.

    Args:
        data_queue (Queue): The queue to store the processed data.

    """
    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_socket.bind(('192.168.1.100', 8080))

    try:
        output_data = np.empty((window_size, 64), dtype=np.float32)
        idx = 0
        while True:
            data, addr = udp_socket.recvfrom(1300)
            transposed_data = np.frombuffer(data[18:1298], dtype='<i2').reshape(10, 64)
            output_data[idx:idx + 10, :] = transposed_data
            idx += 10
            if idx == window_size:
                data_queue.put(output_data)
                event.set()
                idx = 0
    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        udp_socket.close()

def online_td_calc(gestures: list, model_path: str,data_queue: Queue, event: Event, gesture_queue: Queue) -> None:
    """
    Processes data from the queue and predicts gestures using the model.

    Args:
        gestures (list): List of gestures.
        model_path (str): Path to the trained model.
        data_queue (Queue): Queue to get data from.

    Returns:
        None
    """
    cf.model = tccnn_model_creat()
    cf.model.load_weights(model_path)

    while True:
        event.wait()
        if not data_queue.empty():
            data = data_queue.get() / cf.scaling
            filtered_data = bandpass_and_notch_filter(data)
            window_data_feature = tf.convert_to_tensor(calc_td(filtered_data), dtype=tf.float32)
            predictions = cf.model.predict(window_data_feature)
            predicted_class = np.argmax(predictions, axis=1)
            gesture_queue.put([predicted_class])
            event.clear()
        else:
            print("data queue is empty, wait...")
            event.clear()

class DataReceiver(QThread):

    data_received = pyqtSignal(str)

    def __init__(self, gesture_queue: Queue, event: Event):
        super().__init__()
        self.gesture_queue = gesture_queue
        self.event = event

    def run(self):
        while True:
            if not self.gesture_queue.empty():
                data = self.gesture_queue.get()
                self.data_received.emit(data)

class GestureApp(QWidget):
    def __init__(self, gesture_queue: Queue, event: Event):
        super().__init__()

        # title
        self.setWindowTitle("Real-Time Gesture Prediction")
        # window's position and size (x=100, y=100, width=600, height=600)
        self.setGeometry(100, 100, 600, 600)
        # window's background color to light gray
        self.setStyleSheet("background-color: #f4f4f9;")

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # font to Arial, font size 36px
        font = QFont("Arial", 36)
        # Set the font to bold
        font.setBold(True)

        self.gesture_label = QLabel("Predicted gesture: Waiting for input...", self)
        self.gesture_label.setAlignment(Qt.AlignCenter)
        self.gesture_label.setFont(font)
        self.gesture_label.setStyleSheet(""" 
            color: #333;  # Set the font color to dark gray
            border: 2px solid #004d40;  # Set the border to 2px solid dark green
            border-radius: 20px;  # Set the border radius to 20px, adding rounded corners
            padding: 20px;  # Set the label's padding to 20px
            background-color: #b2dfdb;  # Set the background color to light green
        """)

        self.gesture_image = QLabel(self)
        self.gesture_image.setAlignment(Qt.AlignCenter)

        self.layout.addWidget(self.gesture_label)
        self.layout.addWidget(self.gesture_image)

        self.quit_button = QPushButton("Quit", self)
        self.quit_button.setStyleSheet("""  
            background-color: #FF5722;  # Set the button's background color to orange
            color: white;  # Set the button's text color to white
            font-size: 20px;  # Set the button's font size to 20px
            padding: 10px 20px;  # Set the button's padding (10px top/bottom, 20px left/right)
            border-radius: 10px;  # Set the button's border radius to 10px, adding rounded corners
        """)
        self.quit_button.clicked.connect(self.close)

        self.data_receiver = DataReceiver(gesture_queue, event)
        self.data_receiver.data_received.connect(self.update_gesture)
        self.data_receiver.start()

    def update_gesture(self, gesture_prediction: str):
        self.gesture_label.setText(f"Predicted gesture: {gesture_prediction}")
        self.update_image(gesture_prediction)


    def update_image(self, gesture_prediction: str):

        # todo
        if gesture_prediction == "Swipe Left":
            image_path = "images/swipe_left.png"

        else:
            image_path = ""

        if image_path:
            pixmap = QPixmap(image_path)
            # Adjust image size to 300x300 while keeping the aspect ratio
            self.gesture_image.setPixmap(pixmap.scaled(300, 300,Qt.KeepAspectRatio))
        else:
            self.gesture_image.clear()

    def closeEvent(self, event):
        self.data_receiver.terminate()
        event.accept()


if __name__ == "__main__":
    # todo change the gesture list
    gesture_list = [
        "Wave",
        "Fist",
        "Point",
        "Thumb Up",
        "Thumb Down",
        "Peace"
    ]
    cf.config_read()

    data_queue = Queue()
    gesture_queue = Queue()
    event = Event()

    generate_process = Process(
        target=online_data_read,
        args=(data_queue, event, cf.window_size))

    generate_process.start()

    # todo input the model path
    receive_process = Process(
        target=online_td_calc,
        args=(gesture_list,"model_path",data_queue,event,gesture_queue))

    receive_process.start()

    app = QApplication(sys.argv)
    window = GestureApp(gesture_queue, event)
    window.show()
    sys.exit(app.exec_())



