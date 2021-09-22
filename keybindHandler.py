import json
import sys
import os
import stateManager
from pynput.keyboard import Key, Controller as KeyController, Controller as MouseController
from pynput.mouse import Button, Controller as MouseController

def create_action_to_button(config_path: str) -> dict:
    config = json.load(open(config_path))
    # Check for duplicate binds.
    list_config = [config[x] for x in config]
    for x in config:
        if list_config.count(config[x]) > 1:
            print("Error: found a duplicate input being used in config.json.")
            print(f"Please rebind {x} to a key other than {config[x]}")
            stateManager.is_exiting.set()
            #sys.exit()
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

def print_buttons():
    return str([x for x in BUTTON_TO_ACTION])

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
    str(Key.tab).lower(): 'tab',
    str(Key.media_volume_up).lower(): 'media_volume_up',
    str(Key.enter).lower(): 'enter',
}

MOUSE_BUTTONS_RECORDING = {
    str(Button.left).lower(): "lmouse",
    str(Button.right).lower(): "rmouse"
}

# Define the mouse classes
MOUSE_CLASSES = [-1, -3, -5, -8, -10, -15, -20, -30, -60, -100, -200, -300]
MOUSE_CLASSES.extend([-x for x in MOUSE_CLASSES])
MOUSE_CLASSES.append(0)
MOUSE_CLASSES.sort()

MOUSE_CLASS_SIZE = len(MOUSE_CLASSES)

# Create an array of all zeros for every possible button action
EMPTY_BUTTONS_ONEHOT = [0 for _ in range(len(ACTION_TO_BUTTON))]
EMPTY_CLASSES_ONEHOT = [0 for _ in range(len(ACTION_TO_BUTTON))]
EMPTY_CLASSES_ONEHOT.extend([0 for _ in range(len(MOUSE_CLASSES))])
EMPTY_CLASSES_ONEHOT.extend([0 for _ in range(len(MOUSE_CLASSES))])

# Handle Recovery
buttons_pressed_by_code = [0 for _ in ONEHOTINDEX_TO_BUTTON]

# Handle Recording
buttons_pressed_by_human = [0 for _ in ONEHOTINDEX_TO_BUTTON]
last_mouse_moved_x = 0
last_mouse_moved_y = 0


"""Record the given input"""
def record_input(input, pressed: bool):
    pynput_format = str(input).lower()
    if pynput_format in SPECIAL_KEYS_RECORDING:
        if SPECIAL_KEYS_RECORDING[pynput_format] in BUTTON_TO_ACTION:
            index = BUTTON_TO_ONEHOTINDEX[SPECIAL_KEYS_RECORDING[pynput_format]]
            buttons_pressed_by_human[index] = int(pressed)
    elif pynput_format in MOUSE_BUTTONS_RECORDING:
        if MOUSE_BUTTONS_RECORDING[pynput_format] in BUTTON_TO_ACTION:
            index = BUTTON_TO_ONEHOTINDEX[MOUSE_BUTTONS_RECORDING[pynput_format]]
            buttons_pressed_by_human[index] = int(pressed)
    else:
        try:
            pynput_format = str(input).lower().replace("'", "")
            index = BUTTON_TO_ONEHOTINDEX[pynput_format]
            buttons_pressed_by_human[index] = int(pressed)
        except:
            pass

"""A higher order function that returns a function which will map one-hot encoded
actions from configuration file a to a one-hot encoded set of actions from
configuration file b."""
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
        """This works by mapping the index of each element in actions to the btn name for config a,
        which is then fed into the action map for config a which is then fed into the btn map for
        config b which is then fed into the one-hot index mapping for config b which is then used
        to generate the one-hot encoded array for the control scheme. This phyiscally hurt to write."""
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
        if buttons_encoded[i] == 1 and buttons_pressed_by_code[i] == 0:
            if len(button) == 1:
                buttons_pressed_by_code[i] = 1
                keyboard.press(button)
            elif button in SPECIAL_KEYS:
                buttons_pressed_by_code[i] = 1
                keyboard.press(SPECIAL_KEYS[button])
            elif button in MOUSE_BUTTONS:
                buttons_pressed_by_code[i] = 1
                mouse.press(MOUSE_BUTTONS[button])
        # Attempt to release if not already released
        elif buttons_encoded[i] == 0 and buttons_pressed_by_code[i] == 1:
            if len(button) == 1:
                buttons_pressed_by_code[i] = 0
                keyboard.release(button)
            elif button in SPECIAL_KEYS:
                buttons_pressed_by_code[i] = 0
                keyboard.release(SPECIAL_KEYS[button])
            elif button in MOUSE_BUTTONS:
                buttons_pressed_by_code[i] = 0
                mouse.release(MOUSE_BUTTONS[button])

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


"""This function is useful for printing the actual names
of the keys pressed and also can be used for clearing all
pressed keys."""
def get_keys_pressed():
    debug_keys_pressed = []
    for i in range(len(buttons_pressed_by_human)):
        if buttons_pressed_by_human[i] == 1:
            debug_keys_pressed.append(ONEHOTINDEX_TO_BUTTON[i])
    return debug_keys_pressed

"""This function is useful for printing the actual names
of the keys pressed and also can be used for clearing all
pressed keys."""
def get_your_keys_pressed(keys):
    debug_keys_pressed = []
    for i in range(len(keys)):
        if keys[i] == 1:
            debug_keys_pressed.append(ONEHOTINDEX_TO_BUTTON[i])
    return debug_keys_pressed




    
