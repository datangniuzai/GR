#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/3/19 13:05
# @Author : Jason.LI
# @File : robotic_hand.py
# @Software: PyCharm

from typing import Union,List
import warnings

from online.controlled_object.robot_hand import RobotHand
from online.online_demo_sub_base import GestureSubscriberBase


class RobotHandControl(GestureSubscriberBase):
    def __init__(self,hand_prot: str, hand_id: int, gesture_ip: str, gesture_port:int):
        super().__init__(gesture_ip, gesture_port)
        self.hand = RobotHand(port=hand_prot, baudrate=9600, hand_id=hand_id)
        self.GESTURE_MAPPING = {
            1: {"angles": [10, 10, 10, 1000, 10, 10]},
            2: {"angles": [10, 10, 1000, 1000, 10, 10]},
            3: {"angles": [10, 1000, 1000, 1000, 10, 10]},
            4: {"angles": [1000, 1000, 1000, 1000, 10, 10]},
            5: {"angles": [1000, 1000, 1000, 1000, 400, 1000],},
            6: {"angles": [1000, 0, 0, 0, 600, 1000]},
            7: {"angles": [300, 300, 300, 300, 800, 0]},
            8: {"angles": [0, 0, 0, 1000, 1000, 1000]},
            9: {"angles": [0, 0, 0, 600, 300, 800]},
            10: {"angles": [0, 0, 0, 0, 400, 1000]},
            11: {"angles": [200, 200, 200, 200, 1000, 0]},
            12: {"angles": [0, 0, 0, 0, 1000, 1000]},
            13: {"angles": [1000, 1000, 1000, 300, 800, 0],},
            14: {"angles": False},
            15: {"angles": False},
            16: {"angles": [300, 400, 300, 400, 800, 500]},
            17: {"angles": False},
        }
    def execute_control(self, angles_list:List[int]):
        self.hand.set_angles(angles_list)

    def map_gesture_to_control(self, gesture_message: str) -> Union[str, int, None, List[int]]:

        if gesture_message not in self.GESTURE_MAPPING:
            warnings.warn(f"Warning: gesture_message {gesture_message} not found.", RuntimeWarning)
            return []

        return self.GESTURE_MAPPING[gesture_message].get("angles", [])


if __name__ == '__main__':
    robot_hand_control = RobotHandControl("COM3",1,"192.168.1.100",5555)
    robot_hand_control.connect()
    robot_hand_control.run()