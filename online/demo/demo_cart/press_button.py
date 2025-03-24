#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/3/24 14:21
# @Author : Jason.LI
# @File : press_button.py
# @Software: PyCharm

import pyautogui
import pydirectinput
import sys
import time
from pygetwindow import getWindowsWithTitle
pydirectinput.FAILSAFE = False

def active_game_window():
    windows = getWindowsWithTitle("simple64")
    if len(windows) == 0:
        print("Cannot find game window, Please start mupen64 emulator and load game")
        sys.exit()
    w = windows[0]
    w.activate()
    print("success to find the game window.")
    w.maximize()

def enter():
    pydirectinput.press('enter')

def go_straight():
    pyautogui.keyDown('shift')

def stop():
    pyautogui.keyUp('shift')

def left():
    pyautogui.keyDown('left')
    time.sleep(0.1)
    pyautogui.keyUp('left')

def right():
    pyautogui.keyDown('right')
    time.sleep(0.1)
    pyautogui.keyUp('right')

def z():
    pyautogui.press('z')

def up():
    pass

def esc():
    release_all_keys()
    pyautogui.press('esc')

def release_all_keys():
    pyautogui.keyUp('shift')
    pyautogui.keyUp('down')
    pyautogui.keyUp('left')
    pyautogui.keyUp('right')


if __name__ == '__main__':
    active_game_window()
