#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/3/19 15:34
# @Author : Jason.LI
# @File : demo_gesture_display.py
# @Software: PyCharm

import os
import sys

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QPixmap
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton,QHBoxLayout

from online.online_show_base import GestureSubscriberBase
from config import find_project_root

class GestureDisplaySubscriber(GestureSubscriberBase):
    """
    A subclass of GestureSubscriberBase that displays received gestures in a PyQt5 window.
    """

    def __init__(self, ip: str, port: int):
        super().__init__(ip, port)

        self.GESTURE_MAPPING = {
                1: {"name": "gesture1", "image": "data/gesture_pictures/gesture1.png"},
                2: {"name": "gesture2", "image": "data/gesture_pictures/gesture2.png"},
                3: {"name": "gesture3", "image": "data/gesture_pictures/gesture3.png"},
                4: {"name": "gesture4", "image": "data/gesture_pictures/gesture4.png"},
                5: {"name": "gesture5", "image": "data/gesture_pictures/gesture5.png"},
                6: {"name": "gesture6", "image": "data/gesture_pictures/gesture6.png"},
                7: {"name": "gesture7", "image": "data/gesture_pictures/gesture7.png"},
                8: {"name": "gesture8", "image": "data/gesture_pictures/gesture8.png"},
                9: {"name": "gesture9", "image": "data/gesture_pictures/gesture9.png"},
                10: {"name": "gesture10", "image": "data/gesture_pictures/gesture10.png"},
                11: {"name": "gesture11", "image": "data/gesture_pictures/gesture11.png"},
                12: {"name": "gesture12", "image": "data/gesture_pictures/gesture12.png"},
                13: {"name": "gesture13", "image": "data/gesture_pictures/gesture13.png"},
                14: {"name": "gesture14", "image": "data/gesture_pictures/gesture14.png"},
                15: {"name": "gesture15", "image": "data/gesture_pictures/gesture15.png"},
                16: {"name": "gesture16", "image": "data/gesture_pictures/gesture16.png"},
                17: {"name": "gesture17", "image": "data/gesture_pictures/gesture17.png"}
            }

        self.project_root  = find_project_root()

        self.quit_button = None
        self.gesture_label = None
        self.image_label = None
        self.layout = None

        self.app = QApplication(sys.argv)

        self.window = QWidget()

        self.init_ui()

    def init_ui(self):
        """
        Initialize PyQt5 UI to display gesture
        """
        self.window.setWindowTitle("Real-Time Gesture Prediction")
        self.window.setGeometry(100, 100, 600, 600)
        self.window.setStyleSheet("background-color: #f4f4f9;")

        self.layout = QVBoxLayout()
        self.window.setLayout(self.layout)

        font = QFont("Arial", 50)
        font.setBold(True)

        # gesture_label
        self.gesture_label = QLabel("Predicted Gesture: Waiting for Input...", self.window)
        self.gesture_label.setAlignment(Qt.AlignCenter)
        self.gesture_label.setFont(font)
        self.gesture_label.setStyleSheet(
            """ 
            color: #333;  /* Font color: dark gray */
            border: 2px solid #004d40;  /* Border: 2px dark green */
            border-radius: 20px;  /* Border radius: 20px */
            padding: 20px;  /* Padding: 20px */
            background-color: #b2dfdb;  /* Background color: light green */
        """
        )

        # image_label_container
        image_container = QWidget(self.window)
        image_container.setLayout(QHBoxLayout())
        image_container.layout().setAlignment(Qt.AlignCenter)

        # image_label
        self.image_label = QLabel(self.window)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedSize(500, 500)

        # Add the image label to the container
        image_container.layout().addWidget(self.image_label)

        # Add the container to the main layout
        self.layout.addWidget(self.gesture_label)
        self.layout.addWidget(image_container)

        # Quit button
        self.quit_button = QPushButton("Exit", self.window)
        self.quit_button.setStyleSheet(
            """  
            background-color: #FF5722;  /* Background color: orange */
            color: white;  /* Font color: white */
            font-size: 20px;  /* Font size: 20px */
            padding: 10px 20px;  /* Padding: 10px vertically, 20px horizontally */
            border-radius: 10px;  /* Border radius: 10px */
        """
        )
        self.quit_button.clicked.connect(self.window.close)

        self.layout.addWidget(self.quit_button)

        self.window.show()

    def map_gesture_to_control(self, gesture_message: str):
        """
        Maps a gesture message to a control command (in this case, just returns the gesture).
        """

        gesture_info = self.GESTURE_MAPPING.get(int(gesture_message), {"name": "Unknown", "image": ""})

        return gesture_info

    def execute_control(self, control_info):
        """
        Executes the control logic by updating the UI with the received gesture.
        """

        gesture_name = control_info["name"]

        image_path = control_info.get("image", "")

        self.gesture_label.setText(f"Predicted gesture: {gesture_name}")

        image_path = os.path.join(self.project_root ,image_path)

        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            Warning(f"Failed to load image: {image_path}")
        else:
            pixmap = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio)
            self.image_label.setPixmap(pixmap)

        self.app.processEvents()


if __name__ == "__main__":
    subscriber = GestureDisplaySubscriber(ip="localhost", port=5555)
    subscriber.connect()
    subscriber.run()
