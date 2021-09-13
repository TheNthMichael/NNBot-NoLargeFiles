import numpy as np
from threading import Event, Lock
from pynput.keyboard import Key, Controller as KeyController, Listener as KeyListener
from pynput.mouse import Button, Controller as MouseController, Listener as KeyListener
import keybindHandler

# This will be run in another thread outside the main loop so that the main loop can control when frames are pulled
class ActionHandler:
    def __init__(self, fps) -> None:
        self.fps = fps
        self.action_onehot = keybindHandler.EMPTY_ACTIONS_ONEHOT
        self.keyboard = KeyController()
        self.mouse = MouseController()

    def update(self):
        keybindHandler.update_button_presses(self.keyboard, self.mouse, self.action_onehot)

    def set_controller_action(self, actions_onehot):
        self.action_onehot = actions_onehot
        keybindHandler.update_button_presses(self.keyboard, self.mouse, self.action_onehot)

    def release_pressed_buttons(self):
        keybindHandler.release_pressed_buttons(self.keyboard, self.mouse)


def action_handler_thread(action_handler: ActionHandler, exit_event, data_lock=None):
    try:
        while not exit_event.is_set():
            action_handler.update()
    except Exception as e:
        print(e)
        exit_event.set()
    finally:
        action_handler.release_pressed_buttons()
        print("Closing frame_handler_thread...")