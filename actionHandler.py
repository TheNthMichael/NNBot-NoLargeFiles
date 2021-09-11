import subprocess
from typing_extensions import TypeVarTuple
import numpy as np
from threading import Event, Lock
from pynput.keyboard import Key, Controller as KeyController, Listener as KeyListener
import keybindHandler

# This will be run in another thread outside the main loop so that the main loop can control when frames are pulled
class ActionHandler:
    def __init__(self, fps) -> None:
        self.fps = fps
        self.action_onehot = keybindHandler.EMPTY_ACTIONS_ONEHOT
        self.keyboard = KeyController()

    def update(self):
        pass

    def set_controller_action(self, actions_onehot):
        self.action


def frame_handler_thread(frame_handler: ActionHandler, exit_event, data_lock=None):
    try:
        while not exit_event.is_set():
            frame_handler.update()
    except Exception as e:
        print(e)
        exit_event.set()
    finally:
        print("Closing frame_handler_thread...")