import json
import os
from pynput.keyboard import Key, Controller as KeyController, Controller as MouseController
from pynput.mouse import Button, Controller as MouseController

def create_action_to_button(config_path: str) -> dict:
    config = json.load(open(config_path))
    # Verify that this is correct
    assert([type(x) is str and type(config[x]) is str for x in config])
    return config

def create_button_to_onehot(config_path: str) -> dict:
    config = json.load(open(config_path))
    # Verify that this is correct
    assert([type(x) is str and type(config[x]) is str for x in config])
    i = 0
    button2onehot = {}
    for x in config:
        button2onehot[config[x]] = i
        i += 1
    return button2onehot

ACTION_TO_BUTTON = create_action_to_button(".\\Resources\\config.json")
BUTTON_TO_ACTION = {v: k for k, v in ACTION_TO_BUTTON.items()}

BUTTON_TO_ONEHOTINDEX = create_button_to_onehot(".\\Resources\\config.json")
ONEHOTINDEX_TO_BUTTON = {v: k for k, v in BUTTON_TO_ONEHOTINDEX.items()}

ACTION_CLASS_SIZE = len(ONEHOTINDEX_TO_BUTTON)

SPECIAL_KEYS = {
    'alt': Key.alt,
    'space': Key.space,
    'shift': Key.shift,
    'caps_lock': Key.caps_lock,
    'ctrl': Key.ctrl,
    'esc': Key.esc,
    'tab': Key.tab,
    'media_volume_up': Key.media_volume_up,
    'enter': Key.enter
}

MOUSE_BUTTONS = {
    'lmouse': Button.left,
    'rmouse': Button.right
}

SPECIAL_KEYS_RECORDING = {
    str(Key.alt).lower(): 'alt',
    str(Key.space).lower(): 'space',
    str(Key.shift).lower(): 'shift',
    str(Key.caps_lock).lower(): 'caps_lock',
    str(Key.ctrl).lower(): 'ctrl',
    str(Key.esc).lower(): 'esc',
    str(Key.tab).lower(): 'tab',
    str(Key.media_volume_up).lower(): 'media_volume_up',
    str(Key.enter).lower(): 'enter',
}

# Create an array of all zeros for every possible button action
EMPTY_BUTTONS_ONEHOT = [0 for _ in range(len(ACTION_TO_BUTTON))]

# Define the mouse classes
MOUSE_CLASSES = [-1, -3, -5, -10, -20, -30, -60, -100, -200, -300]
MOUSE_CLASSES.extend([-x for x in MOUSE_CLASSES])
MOUSE_CLASSES.append(0)
MOUSE_CLASSES.sort()

MOUSE_CLASS_SIZE = len(MOUSE_CLASSES)

# Handle Recovery
buttons_pressed_by_code = [0 for _ in ONEHOTINDEX_TO_BUTTON]

# Handle Recording
buttons_pressed_by_human = [0 for _ in ONEHOTINDEX_TO_BUTTON]

def control_scheme_transform(config_path_a, config_path_b):
    act_2_btn_a = create_action_to_button(config_path_a)
    btn_2_act_a = {v: k for k, v in act_2_btn_a.items()}
    btn_2_oh_a = create_button_to_onehot(config_path_a)
    oh_2_btn_a = {v: k for k, v in btn_2_oh_a.items()}

    act_2_btn_b = create_action_to_button(config_path_b)
    btn_2_oh_b = create_button_to_onehot(config_path_b)

    assert(len(act_2_btn_a) == len(act_2_btn_b))

    """Takes in a one-hot encoded set of buttons in scheme a to a one-hot encoded set of
    in scheme b by each configs mapping to the universal action names in config.json."""
    def transform(actions):
        # convert a's actions from one-hot to buttons
        """a_btn = [oh_2_btn_a[x] for x in range(len(actions))]
        a_act = [btn_2_act_a[x] for x in a_btn]

        b_btn = [act_2_btn_b[x] for x in a_act]
        b_oh = [btn_2_oh_b[x] for x in b_btn]"""
        actions_b = [x for x in actions]
        for i in range(len(actions)):
            actions_b[btn_2_oh_b[act_2_btn_b[btn_2_act_a[oh_2_btn_a[i]]]]] = actions[i]
        return actions_b
    return transform
        

"""Split the one-hot encoded output to its separate one-hot encoded classes.

Returns a tuple containing the keys_onehot, mousex_onehot, and mousey_onehot arrays."""
def output_to_mappings(output):
    keys_f = output[:ACTION_CLASS_SIZE]
    mousex_f = output[ACTION_CLASS_SIZE:ACTION_CLASS_SIZE+MOUSE_CLASS_SIZE]
    mousey_f = output[ACTION_CLASS_SIZE+MOUSE_CLASS_SIZE:ACTION_CLASS_SIZE+MOUSE_CLASS_SIZE+MOUSE_CLASS_SIZE]
    return keys_f, mousex_f, mousey_f


"""Transform the two-tuple of floats for mouse movement into two one-hot encoded arrays for x and y movement.

This function looks for the closest class to convert the floating point input into by checking for the minimum
distance between the original mouse and each element in the one-hot encoded class of mouse movement values.

Returns a tuple containing the one-hot encoded arrays for mousex and mousey."""
def mouse_to_classification(mouse):
        assert(len(MOUSE_CLASSES) != 0)
        mousex_class_output = [0 for _ in MOUSE_CLASSES]
        mousey_class_output = [0 for _ in MOUSE_CLASSES]
        indx = 0
        indy = 0
        for i in range(len(MOUSE_CLASSES)):
            if abs(mouse[0] - MOUSE_CLASSES[i]) <= abs(mouse[0] - MOUSE_CLASSES[indx]):
                indx = i
            if abs(mouse[1] - MOUSE_CLASSES[i]) <= abs(mouse[1] - MOUSE_CLASSES[indy]):
                indy = i
        mousex_class_output[indx] = 1        
        mousey_class_output[indy] = 1    
        mousex_class_output.extend(mousey_class_output)
        return mousex_class_output

"""Takes the one-hot encoded buttons and either presses that button down or releases that button.
Manages an array of currently pressed keys for release when we stop the bot.
"""
def update_button_presses(keyboard, mouse, buttons_encoded):
    for i in range(len(buttons_encoded)):
        button = ONEHOTINDEX_TO_BUTTON[i]
        # Attempt to press if not already pressed
        if buttons_encoded[i] == 1 and buttons_pressed_by_code[i] != 1:
            if len(button) == 1:
                # keypress
                keyboard.press(button)
            elif button in SPECIAL_KEYS:
                keyboard.press(SPECIAL_KEYS[button])
            elif button in MOUSE_BUTTONS:
                mouse.press(MOUSE_BUTTONS[button])
            # Record button
            buttons_pressed_by_code[i] = 1
        # Attempt to release if not already released
        elif buttons_pressed_by_code[i] != 0:
            if len(button) == 1:
                # keypress
                keyboard.release(button)
            elif button in SPECIAL_KEYS:
                keyboard.release(SPECIAL_KEYS[button])
            elif button in MOUSE_BUTTONS:
                mouse.release(MOUSE_BUTTONS[button])
            # Record button    
            buttons_pressed_by_code[i] = 0

"""Takes in your controllers and releases any buttons pressed on them.
"""
def release_pressed_buttons(keyboard, mouse):
    for i in range(len(buttons_pressed_by_code)):
        button = ONEHOTINDEX_TO_BUTTON[i]
        # Attempt to press if not already pressed
        if buttons_pressed_by_code[i] == 1:
            if len(button) == 1:
                # keypress
                keyboard.release(button)
            elif button in SPECIAL_KEYS:
                keyboard.release(SPECIAL_KEYS[button])
            elif button in MOUSE_BUTTONS:
                mouse.release(MOUSE_BUTTONS[button])
            # Record button
            buttons_pressed_by_code[i] = 0




    
