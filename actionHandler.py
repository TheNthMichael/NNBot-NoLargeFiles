import ctypes
from math import ceil, floor
import time
import traceback
import numpy as np
from threading import Event, Lock
from pynput.keyboard import Key, Controller as KeyController, Listener as KeyListener
from pynput.mouse import Button, Controller as MouseController, Listener as KeyListener
import keybindHandler

""" Below is code for an attempted smooth move function, however it tends to move more than the designated number of pixels.
        >>> timeBeginPeriod = ctypes.windll.winmm.timeBeginPeriod
        >>> timeBeginPeriod(1)
        def smooth(x,y,t):
    ...     distance = (x**2 + y**2)**0.5
    ...     step_count = int(ceil(distance))
    ...     interval = t / distance
    ...     relx = 0
    ...     rely = 0
    ...     realx = 0
    ...     realy = 0
    ...     for step in range(step_count):
    ...             relx += x * (1 / step_count)
    ...             rely += y * (1 / step_count)
    ...             if relx >= 1:
    ...                     realx = floor(relx)
    ...                     relx -= realx
    ...             if rely >= 1:
    ...                     realy = floor(rely)
    ...                     rely -= realy
    ...             ctypes.windll.user32.mouse_event(0x01, int(realx), int(realy), 0, 0)
    ...             realx = 0
    ...             realy = 0
    ...             time.sleep(t * (1 / step_count))
...
>>> smooth(10,10,1/20)
>>> smooth(10,10,1/20)
>>> smooth(10,10,1/20)
>>> smooth(100,100,1/20)
>>> from ctypes import windll
>>> import ctypes
>>> timeBeginPeriod = ctypes.windll.winmm.timeBeginPeriod
>>> timeBeginPeriod(1)
0
>>> smooth(100,100,1/20)
>>> smooth(100,100,1/20)
>>> smooth(10,10,1/20)
>>> smooth(50,50,1/20)
>>> smooth(50,50,1/20)
"""
"""This class handles the movement movement actions of the bot.

The class uses ctypes for mouse move events and pynput for keyboard and mouse button presses."""
class ActionHandler:
    def __init__(self, fps) -> None:
        # reset timer otherwise enjoy massive delays with time.sleep() I cannot stress the importance of this.
        timeBeginPeriod = ctypes.windll.winmm.timeBeginPeriod
        timeBeginPeriod(1)
        self.fps = fps
        self.action_onehot = keybindHandler.EMPTY_BUTTONS_ONEHOT 
        self.mouse_x = 0
        self.mouse_y = 0
        self.relx = 0
        self.rely = 0
        self.realx = 0
        self.realy = 0
        self.distance = 0
        self.step_count = 0.000001
        self.step = 0
        self.interval = 0
        self.keyboard = KeyController()
        self.mouse = MouseController()

    """Updates the button presses (expected to change every 1/20 seconds) and smoothly transitions the mouse over this time frame.
    This update function requires that the timer is reset to avoid time.sleep delays growing larger.
    This smooth mouse movement relies on making smaller increments < 1 pixel wait until a whole pixel of transition is reached before doing any action.
    This may have issues with precision as the value must go over the integer threshold to be of use. It may be useful to overestimate pixel coords."""
    def update(self):
        # handle logic for smooth mouse movement
        self.relx += self.mouse_x * (1 / self.step_count)
        self.rely += self.mouse_y * (1 / self.step_count)
        if self.relx >= 1:
            self.realx = floor(self.relx)
            self.relx -= self.realx
        if self.rely >= 1:
            self.realy = floor(self.rely)
            self.rely -= self.realy
        # Move mouse by measured real x and y values.
        ctypes.windll.user32.mouse_event(0x01, self.realx, self.realy, 0, 0)
        self.realx = 0
        self.realy = 0
        self.step += 1

    """Set the actions for the actionHandler to update over the span of 1/fps.
    
    Button presses are passed in one-hot format and mouse movement is passed
    as dx, dy where the change is for the duration of a frame aka 1/fps."""
    def set_controller_action(self, actions_onehot, mousex, mousey):
        self.action_onehot = actions_onehot
        self.mouse_x = mousex
        self.mouse_y = mousey
        self.distance = (self.mouse_x**2 + self.mouse_y**2)**0.5
        self.step_count = int(ceil(self.distance))
        self.step = 0
        if (self.step_count == 0):
            self.step_count = 0.0000001
            self.interval = 0
        else:
            self.interval = 1 / (self.fps * self.step_count)
        self.relx = 0
        self.rely = 0
        self.realx = 0
        self.realy = 0
        # press keybinds before moving mouse to avoid mouse delays
        keybindHandler.update_button_presses(self.keyboard, self.mouse, self.action_onehot)
        # start updating
        self.update()

    def release_pressed_buttons(self):
        keybindHandler.release_pressed_buttons(self.keyboard, self.mouse)

"""A thread for running the Mouse and Keyboard I/O.

Mouse inputs are interpolated over the expected fps.
This may lead to slightly inaccurate movements.
This can be fixed by creating your ActionHandler with an fps higher than the expected fps."""
def action_handler_thread(action_handler: ActionHandler, exit_event, recording_event):
    try:
        while not exit_event.is_set():
            # Here we disable updating.
            if recording_event.is_set():
                action_handler.update()
                time.sleep(action_handler.interval)
            else:
                action_handler.release_pressed_buttons()
                time.sleep(1/action_handler.fps)
    except Exception as e:
        traceback.print_exc()
        print(f"Error in ActionThread: {e}")
    finally:
        print("Closing action_handler_thread...")
        action_handler.release_pressed_buttons()
        exit_event.set()