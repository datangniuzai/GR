#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/3/19 13:08
# @Author : Jason.LI
# @File : robot_hand.py
# @Software: PyCharm

import serial
import logging
from typing import List, Optional


class RobotHand:
    """
    A class to control a robotic hand via serial communication.
    """

    def __init__(self, port: str, baudrate: int = 9600, timeout: float = 1.0, hand_id: int = 1):
        """
        Initializes the RobotHand.

        Args:
            port (str): The serial port to connect to (e.g., 'COM3' or '/dev/ttyUSB0').
            baudrate (int): The baud rate for serial communication (default: 9600).
            timeout (float): The timeout for serial communication (default: 1.0).
            hand_id (int): The ID of the robotic hand (default: 1).
        """
        self.hand_id = hand_id
        self.ser = None  # Initialize serial connection as None
        self._connect(port, baudrate, timeout)

    def __del__(self):
        """
        Closes the serial connection when the object is destroyed.
        """
        if self.ser and self.ser.is_open:
            self.ser.close()
            logging.info("Serial connection closed.")

    def _connect(self, port: str, baudrate: int, timeout: float):
        """
        Connects to the robotic hand via serial communication.

        Args:
            port (str): The serial port to connect to.
            baudrate (int): The baud rate for serial communication.
            timeout (float): The timeout for serial communication.
        """
        try:
            self.ser = serial.Serial(port=port, baudrate=baudrate, timeout=timeout)
            logging.info(f"Connected to robotic hand on port {port}.")
        except serial.SerialException as e:
            logging.error(f"Failed to connect to robotic hand: {e}")
            raise

    @staticmethod
    def data2bytes(data: int) -> List[int]:
        """
        Converts a number to a list of two bytes.

        Args:
            data (int): The number to convert.

        Returns:
            List[int]: A list of two bytes.
        """
        rdata = [0xFF] * 2
        if data == -1:
            rdata[0] = 0xFF
            rdata[1] = 0xFF
        else:
            rdata[0] = data & 0xFF
            rdata[1] = (data >> 8) & 0xFF
        return rdata

    @staticmethod
    def num2str(num: int) -> bytes:
        """
        Converts a number to a hexadecimal string and then to bytes.

        Args:
            num (int): The number to convert.

        Returns:
            bytes: The converted bytes.
        """
        hex_str = hex(num)[2:].zfill(2)
        return bytes.fromhex(hex_str)

    @staticmethod
    def checknum(data: List[int], length: int) -> int:
        """
        Calculates a checksum for the data.

        Args:
            data (List[int]): The data to calculate the checksum for.
            length (int): The length of the data to include in the checksum.

        Returns:
            int: The checksum value.
        """
        return sum(data[2:length]) & 0xFF

    @staticmethod
    def check_and_adjust_values(values_list: List[int]) -> List[int]:
        """
        Checks if the values in the list are within the range [0, 1000].
        If not, adjusts the values to be within the range and logs a warning.

        Args:
            values_list (List[int]): A list of integers to check and adjust.

        Returns:
            List[int]: A list of adjusted values.
        """
        adjusted_values = [max(0, min(1000, value)) for value in values_list]

        for value in values_list:
            if value < 0:
                logging.warning(f"Value {value} is less than 0. Adjusted to 0.")
            elif value > 1000:
                logging.warning(f"Value {value} is greater than 1000. Adjusted to 1000.")
        return adjusted_values

    def set_angles(self, angle_list: List[int]) -> Optional[bytes, None]:
        """
        Sets the angles of the robotic hand and sends the data via serial communication.

        Args:
             angle_list (List[int]): A list of angles to set.

        Returns:
            Optional[bytes]: The received data from the serial device, or None if communication fails.
        """
        try:

            angle_list = self.check_and_adjust_values(angle_list)

            datanum = 0x0F
            b = [0] * (datanum + 5)

            # Header and metadata
            b[0] = 0xEB
            b[1] = 0x90
            b[2] = self.hand_id
            b[3] = datanum
            b[4] = 0x12
            b[5] = 0xCE
            b[6] = 0x05

            for i, angle in enumerate(angle_list):
                angle_bytes = self.data2bytes(angle)
                b[7 + 2 * i] = angle_bytes[0]
                b[8 + 2 * i] = angle_bytes[1]

            b[19] = self.checknum(b, datanum + 4)

            put_data = b""
            for i in range(len(b)):
                put_data += self.num2str(b[i])

            self.ser.write(put_data)
            received_data = self.ser.read(9)

            return received_data

        except serial.SerialException as e:
            print(f"Serial communication failed: {e}")
            return None


if __name__ == "__main__":

    hand = RobotHand(port="COM3", baudrate=9600)

    angles = [90, 45, 30, 60, 75]
    hand.set_angles(angles)


