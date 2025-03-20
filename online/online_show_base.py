#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/3/19 11:31
# @Author : Jason.LI
# @File : online_show_base.py
# @Software: PyCharm

import zmq
from abc import ABC, abstractmethod
from typing import Union, List


class GestureSubscriberBase(ABC):
    """
    A class to subscribe to a ZeroMQ publisher and map received gestures to control commands.
    """

    def __init__(self, ip: str, port: int):
        """
        Initializes the GestureSubscriberBase.

        Args:
            ip (str): The IP address of the publisher (e.g., "192.168.1.100" or "localhost").
            port (int): The port number of the publisher (e.g., 5555).
        """
        self.ip = ip
        self.port = port
        self.context = zmq.Context()
        self.zmq_socket = self.context.socket(zmq.SUB)

    def connect(self):
        """
        Connects to the ZeroMQ publisher and starts receiving messages.
        """
        connect_address = f"tcp://{self.ip}:{self.port}"
        self.zmq_socket.connect(connect_address)
        self.zmq_socket.setsockopt_string(zmq.SUBSCRIBE, "")
        print(f"Subscriber connected to {connect_address}")

    def run(self):
        """
        Starts the subscriber loop to receive and process messages and execute control.
        """
        try:
            while True:
                gesture_message = self.zmq_socket.recv_string()
                # print(f"Received gesture: {gesture_message}")

                control_command = self.map_gesture_to_control(gesture_message)
                # print(f"Mapped control command: {control_command}")

                self.execute_control(control_command)
        except KeyboardInterrupt:
            print("Subscriber interrupted.")
        finally:
            self.zmq_socket.close()
            self.context.term()
            print("Subscriber stopped.")

    @abstractmethod
    def map_gesture_to_control(self, gesture_message: str):
        """
        Maps a gesture message to a control command.
        This method must be implemented by subclasses.

        Args:
            gesture_message (str): The received gesture message.

        Returns:
            Union[str, int, None]: The mapped control command, or None if no mapping exists.
        """
        pass

    @abstractmethod
    def execute_control(self, control_command):
        """
        Executes the control logic based on the mapped control command.
        This method must be implemented by subclasses.

        Args:
            control_command: The control command to execute.
        """
        pass


if __name__ == "__main__":
    subscriber = GestureSubscriberBase(ip="192.168.1.100", port=5555)
    subscriber.connect()
    subscriber.run()
