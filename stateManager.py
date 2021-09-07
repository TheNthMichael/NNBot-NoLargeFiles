from PIL.Image import new
import dataEncoder

"""Some things need to be stored globally
for this application to work due to pynput."""

keys_pressed = [0 for x in dataEncoder.KEY_TO_CODE_MAP]

keys_frame_count = [0 for x in dataEncoder.KEY_TO_CODE_MAP]

last_mousex: float = 0
last_mousey: float = 0

last_frame_mousex: float = 0
last_frame_mousey: float = 0

is_recording = False
is_not_exiting = True

screen_cap_scale = 8

monitor_region = {'top': 0, 'left': 0, 'width': 1920, 'height': 1080}

screen_cap_sizes = ((monitor_region['width'] - monitor_region['left']) // screen_cap_scale, (monitor_region['height'] - monitor_region['top']) // screen_cap_scale)

screen_cap_resolution = screen_cap_sizes[0] * screen_cap_sizes[1]

let_go_of_my_keys_please = [0 for x in dataEncoder.KEY_TO_CODE_MAP]

def update_pressed_keys(new_keys):
    for i in range(len(new_keys)):
        if new_keys[i] == 1:
            let_go_of_my_keys_please[i] = 1
        elif new_keys[i] == 0:
            let_go_of_my_keys_please[i] = 0

def let_go_of_pressed_keys(controller):
    for i in range(len(let_go_of_my_keys_please)):
        key = dataEncoder.CODE_TO_KEY_MAP[i]
        if let_go_of_my_keys_please[i] == 1:
            controller.release(key)
            let_go_of_my_keys_please[i] = 0

def update_keys_frame_count():
        assert(len(keys_frame_count) == len(keys_pressed))
        for i in range(len(keys_frame_count)):
            if keys_pressed[i] == 0:
                keys_frame_count[i] = 0
            else:
                keys_frame_count[i] += 1
        return keys_frame_count

def try_add_key_pressed(key):
    try:
        code = dataEncoder.map_key_to_code(key)
        if keys_pressed[code] == 0:
            keys_pressed[code] = 1
    except:
        pass

def try_remove_key_pressed(key):
    try:
        code = dataEncoder.map_key_to_code(key)
        if keys_pressed[code] == 1:
            keys_pressed[code] = 0
    except:
        pass

"""This function is useful for printing the actual names
of the keys pressed and also can be used for clearing all
pressed keys."""
def get_keys_pressed():
    debug_keys_pressed = []
    for i in range(len(keys_pressed)):
        if keys_pressed[i] == 1:
            debug_keys_pressed.append(dataEncoder.map_code_to_key(i))
    #debug_keys_pressed = [dataEncoder.map_code_to_key(x) if keys_pressed[x] else "-" for x in range(len(keys_pressed))].remove("-")
    return debug_keys_pressed

"""This function is useful for printing the actual names
of the keys pressed and also can be used for clearing all
pressed keys."""
def get_your_keys_pressed(keys):
    debug_keys_pressed = []
    for i in range(len(keys)):
        if keys[i] == 1:
            debug_keys_pressed.append(dataEncoder.map_code_to_key(i))
    #debug_keys_pressed = [dataEncoder.map_code_to_key(x) if keys_pressed[x] else "-" for x in range(len(keys_pressed))].remove("-")
    return debug_keys_pressed

