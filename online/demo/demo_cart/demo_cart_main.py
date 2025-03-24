#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/3/24 13:49
# @Author : Jason.LI
# @File : demo_cart_main.py
# @Software: PyCharm

import warnings
from press_button import left, right, z, enter, esc, active_game_window,go_straight,up,release_all_keys,stop
from online.online_demo_sub_base import GestureSubscriberBase
import time

class GestureControlCart(GestureSubscriberBase):
    def __init__(self, ip: str, port: int):
        """
        Initializes the GestureControlCart with IP and port.
        """
        super().__init__(ip, port)

        # Gesture to control mapping
        self.GESTURE_MAPPING = {
            1: {"name": "gesture1", "command": left},
            2: {"name": "gesture2", "command": right},
            3: {"name": "gesture3", "command": z},
            4: {"name": "gesture4", "command": enter},
            5: {"name": "gesture5", "command": esc},
            6: {"name": "gesture6", "command": up},
            7: {"name": "gesture7", "command": go_straight},
            8: {"name": "gesture8", "command": stop},
        }


    def execute_control(self, command):
        """
        Executes the given control command.

        Args:
            command (Callable): A callable function to be executed.
        """
        try:
            command()
            print("send command",command)
        except Exception as e:
            warnings.warn(f"Failed to execute command: {e}", RuntimeWarning)

    def map_gesture_to_control(self, gesture_message: str) -> str:
        """
        Maps the given gesture message to the corresponding control command.

        Args:
            gesture_message (str): The gesture message to map (e.g., "1" for "gesture1").

        Returns:
            str: The command mapped to the gesture (e.g., "left", "right", etc.).
        """
        try:
            gesture_id = int(gesture_message)
            if gesture_id not in self.GESTURE_MAPPING:
                warnings.warn(f"Warning: gesture_message {gesture_message} not found.", RuntimeWarning)
                return "up"
            return self.GESTURE_MAPPING[gesture_id].get("command", "up")
        except ValueError:
            warnings.warn(f"Invalid gesture_message format: {gesture_message}. Expected an integer.", RuntimeWarning)
            return "up"


if __name__ == "__main__":
    try:
        active_game_window()
        time.sleep(1)
        controller = GestureControlCart(ip="localhost", port=5555)
        controller.connect()
        controller.run()
    finally:
        release_all_keys()
