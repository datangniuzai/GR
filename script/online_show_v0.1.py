#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/3/6 11:34
# @Author : Jason.LI
# @File : online_show_v0.1.py
# @Software: PyCharm

import threading
import os
import sys
import serial
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

global hand_id
hand_id = 1

GESTURE_MAPPING = {
    1: {"name": "gesture1", "image": "data/gesture_pictures/gesture1.png", "angles": [10, 10, 10, 1000, 10, 10]},
    2: {"name": "gesture2", "image": "data/gesture_pictures/gesture2.png", "angles": [10, 10, 1000, 1000, 10, 10]},
    3: {"name": "gesture3", "image": "data/gesture_pictures/gesture3.png", "angles": [10, 1000, 1000, 1000, 10, 10]},
    4: {"name": "gesture4", "image": "data/gesture_pictures/gesture4.png", "angles": [1000, 1000, 1000, 1000, 10, 10]},
    5: {"name": "gesture5", "image": "data/gesture_pictures/gesture5.png", "angles": [1000, 1000, 1000, 1000, 400, 1000]},
    6: {"name": "gesture6", "image": "data/gesture_pictures/gesture6.png", "angles": [1000, 0, 0, 0, 600, 1000]},
    7: {"name": "gesture7", "image": "data/gesture_pictures/gesture7.png", "angles": [300, 300, 300, 300, 800, 0]},
    8: {"name": "gesture8", "image": "data/gesture_pictures/gesture8.png", "angles": [0, 0, 0, 1000, 1000, 1000]},
    9: {"name": "gesture9", "image": "data/gesture_pictures/gesture9.png", "angles": [0, 0, 0, 600, 300, 800]},
    10: {"name": "gesture10", "image": "data/gesture_pictures/gesture10.png", "angles": [0, 0, 0, 0, 400, 1000]},
    11: {"name": "gesture11", "image": "data/gesture_pictures/gesture11.png", "angles": [200, 200, 200, 200, 1000, 0]},
    12: {"name": "gesture12", "image": "data/gesture_pictures/gesture12.png", "angles": [0, 0, 0, 0, 1000, 1000]},
    13: {"name": "gesture13", "image": "data/gesture_pictures/gesture13.png", "angles": [1000, 1000, 1000, 300, 800, 0]},
    14: {"name": "gesture14", "image": "data/gesture_pictures/gesture14.png", "angles": False},
    15: {"name": "gesture15", "image": "data/gesture_pictures/gesture15.png", "angles": False},
    16: {"name": "gesture16", "image": "data/gesture_pictures/gesture16.png", "angles": [300, 400, 300, 400, 800, 500]},
    17: {"name": "gesture17", "image": "data/gesture_pictures/gesture17.png", "angles": False},
}

def udp_receiver(data_queue, window_size):
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
                data_queue.put(output_data.copy())
                idx = 0
    except Exception as e:
        print(f"Error in receiver: {e}")
    finally:
        udp_socket.close()

def model_predictor(data_queue, gesture_to_show_queue, model_path):
    model = tccnn_model_creat()
    model.load_weights(model_path)
    while True:
        if not data_queue.empty():
            output_data = data_queue.get()
            output_data = output_data / cf.scaling
            filtered_data = bandpass_and_notch_filter(output_data)
            window_data_feature = tf.expand_dims(tf.convert_to_tensor(calc_td(filtered_data), dtype=tf.float32), axis=0)
            predictions = model.predict(window_data_feature)
            predicted_class = np.argmax(predictions, axis=1)
            gesture_to_show_queue.put([int(predicted_class[0])])

class GestureApp(QWidget):
    def __init__(self, gesture_queue: Queue, event: Event):
        super().__init__()
        self.ser = serial.Serial('COM1',115200)
        print(self.ser)
        self.setWindowTitle("Real-Time Gesture Prediction")
        self.setGeometry(100, 100, 600, 600)
        self.setStyleSheet("background-color: #f4f4f9;")

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        font = QFont("Arial", 36)
        font.setBold(True)

        self.gesture_label = QLabel("Predicted gesture: Waiting for input...", self)
        self.gesture_label.setAlignment(Qt.AlignCenter)
        self.gesture_label.setFont(font)
        self.gesture_label.setStyleSheet(""" 
            color: #333;  /* 字体颜色：深灰色 */
            border: 2px solid #004d40;  /* 边框：2px 深绿色 */
            border-radius: 20px;  /* 圆角：20px */
            padding: 20px;  /* 内边距：20px */
            background-color: #b2dfdb;  /* 背景颜色：浅绿色 */
        """)

        self.gesture_image = QLabel(self)
        self.gesture_image.setAlignment(Qt.AlignCenter)

        self.layout.addWidget(self.gesture_label)
        self.layout.addWidget(self.gesture_image)

        self.quit_button = QPushButton("Quit", self)
        self.quit_button.setStyleSheet("""  
            background-color: #FF5722;  /* 背景颜色：橙色 */
            color: white;  /* 字体颜色：白色 */
            font-size: 20px;  /* 字体大小：20px */
            padding: 10px 20px;  /* 内边距：10px 上下，20px 左右 */
            border-radius: 10px;  /* 圆角：10px */
        """)
        self.quit_button.clicked.connect(self.close)

        self.data_receiver = DataReceiver(gesture_queue, event)
        self.data_receiver.data_received.connect(self.update_gesture)
        self.data_receiver.start()

    def update_gesture(self, gesture_number: int):
        gesture_info = GESTURE_MAPPING.get(gesture_number, {"name": "Unknown", "image": ""})
        gesture_name = gesture_info["name"]
        image_path = gesture_info["image"]
        angles = gesture_info["angles"]
        self.gesture_label.setText(f"Predicted gesture: {gesture_name}")
        self.update_image(image_path,angles)

    def update_image(self, image_path: str,angles):
        if image_path:
            pixmap = QPixmap(image_path)
            self.gesture_image.setPixmap(pixmap.scaled(300, 300, Qt.KeepAspectRatio))
        if angles:
            self.set_angle(*angles)
        else:
            self.gesture_image.clear()

    def closeEvent(self, event):
        self.data_receiver.terminate()
        event.accept()

    def data2bytes(self,data):
        rdata = [0xff]*2
        if data == -1:
            rdata[0] = 0xff
            rdata[1] = 0xff
        else:
            rdata[0] = data&0xff
            rdata[1] = (data>>8)&(0xff)
        return rdata

    def num2str(self,num):
        str = hex(num)
        str = str[2:4]
        if(len(str) == 1):
            str = '0'+ str
        str = bytes.fromhex(str)
        #print(str)
        return str

    def checknum(self,data,leng):
        result = 0
        for i in range(2,leng):
            result += data[i]
        result = result&0xff
        return result

    def set_angle(self, angle1, angle2, angle3, angle4, angle5, angle6):
        global hand_id
        datanum = 0x0F
        b = [0] * (datanum + 5)
        # 包头
        b[0] = 0xEB
        b[1] = 0x90

        # hand_id号
        b[2] = hand_id

        # 数据个数
        b[3] = datanum

        # 写操作
        b[4] = 0x12

        # 地址
        b[5] = 0xCE
        b[6] = 0x05

        # 数据
        b[7] = self.data2bytes(angle1)[0]
        b[8] = self.data2bytes(angle1)[1]

        b[9] = self.data2bytes(angle2)[0]
        b[10] = self.data2bytes(angle2)[1]

        b[11] = self.data2bytes(angle3)[0]
        b[12] = self.data2bytes(angle3)[1]

        b[13] = self.data2bytes(angle4)[0]
        b[14] = self.data2bytes(angle4)[1]

        b[15] = self.data2bytes(angle5)[0]
        b[16] = self.data2bytes(angle5)[1]

        b[17] = self.data2bytes(angle6)[0]
        b[18] = self.data2bytes(angle6)[1]

        b[19] = self.checknum(b, datanum + 4)

        putdata = b''

        for i in range(1, datanum + 6):
            putdata = putdata + self.num2str(b[i - 1])
        self.ser.write(putdata)
        # print('发送的数据：')
        # for i in range(1, datanum + 6):
            # print(hex(putdata[i - 1]))

        getdata = self.ser.read(9)
        # print(type(getdata))
        # print('返回的数据：')
        # for i in range(1,10):
        #     print(getdata[i-1])


class DataReceiver(QThread):
    data_received = pyqtSignal(int)

    def __init__(self, gesture_queue: Queue, event: Event):
        super().__init__()
        self.gesture_queue = gesture_queue
        self.event = event

    def run(self):
        while True:
            if not self.gesture_queue.empty():
                data = self.gesture_queue.get()
                self.data_received.emit(data[0])
                self.event.clear()


if __name__ == "__main__":
    cf.config_read()

    sEMG_data_queue = Queue()
    gesture_queue = Queue()
    event = Event()
    model_path = "data/240909-LJX-Man-S-17/all_train_info/tccnn_fold1_03-03_21-35/models/model_03.keras"
    receiver_thread = threading.Thread(target=udp_receiver, args=(sEMG_data_queue, cf.window_size))
    receiver_thread.start()

    predictor_thread = threading.Thread(
        target=model_predictor,
        args=(sEMG_data_queue, gesture_queue,model_path)
    )
    predictor_thread.start()

    app = QApplication(sys.argv)
    window = GestureApp(gesture_queue, event)
    window.show()
    sys.exit(app.exec_())