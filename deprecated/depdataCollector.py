import numpy as np
import dataEncoder
import cv2
import time
import os
import stateManager
from datetime import datetime
from mss import mss
from PIL import Image
import pynput
from pynput.keyboard import Key, Listener as KeyListener
from pynput.mouse import Listener as MouseListener

def on_press_handler(key):
    stateManager.try_add_key_pressed(key)
    try:
        print('alphanumeric key {0} pressed'.format(key.char))
          
    except AttributeError:
        print('special key {0} pressed'.format(key))

def on_release_handler(key):
    if key == Key.f3:
        stateManager.is_recording = not stateManager.is_recording
        return stateManager.is_not_exiting
    stateManager.try_remove_key_pressed(key)
    print('{0} released'.format(key))
    return stateManager.is_not_exiting

def on_move_handler(x, y):
    stateManager.last_mousex = x
    stateManager.last_mousey = y
    print('Pointer moved to {0}'.format(
        (x, y)))
    return stateManager.is_not_exiting

class TrainingExample:
    def __init__(self, prev_image, prev_inputs, image, \
        inputs, input_frame_count):
        self.prev_image = prev_image
        self.prev_inputs = prev_inputs
        self.image = image
        self.inputs = inputs
        self.input_frame_count = input_frame_count

class DataCollector:
    def __init__(self, dataset_path:str=None) -> None:
        self.dataset_path = dataset_path
        if self.dataset_path == None:
            dirname = os.path.dirname(__file__)
            testdir = os.path.join(dirname, "TestData")
            if not os.path.exists(testdir):
                os.mkdir(testdir)
            
            self.dataset_path = os.path.join(testdir, \
                f"dataset{datetime.now().strftime('%Y%m%d%H%M%S')}")
        os.mkdir(self.dataset_path)
        self.sct = mss()
    
    def write_state_to_output(self, sampleName, frame, mousedx,\
        mousedy, keys_pressed, input_history_flattened):
        file = os.path.join(self.dataset_path, sampleName)
        np.savez(file, image=frame, keys=np.array(keys_pressed),\
            mouse=np.array([mousedx, mousedy]),\
                inputhistory=np.array(input_history_flattened))

    def run(self):
        resize_tuple = (1920 //  stateManager.screen_cap_scale, 1080 //  stateManager.screen_cap_scale)
        resolution = resize_tuple[0] * resize_tuple[1]
        # used to record the time when we processed last frame
        prev_frame_time = 0
        # used to record the time at which we processed current frame
        new_frame_time = 0
        sampleNum = 0
        prev_mousex = None
        prev_mousey = None

        cv2.namedWindow('collector', cv2.WINDOW_KEEPRATIO)

        # Generate an array of empty histories
        output_template = [0 for _ in dataEncoder.KEY_TO_CODE_MAP]
        output_template.extend([0, 0])
        input_history = [output_template[:] for _ in range(dataEncoder.HISTORY_LENGTH)]
        prev_input = []
        with KeyListener(on_press = on_press_handler,
              on_release = on_release_handler) as key_listener:
            with MouseListener(on_move = on_move_handler) as mouse_listener:
                while stateManager.is_not_exiting:
                    self.sct.get_pixels(stateManager.monitor_region)
                    img = Image.frombytes('RGB', (self.sct.width, self.sct.height), self.sct.image)
                    img = np.array(img)
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                    img = cv2.resize(img, resize_tuple)

                    cur_mousex = stateManager.last_mousex
                    cur_mousey = stateManager.last_mousey

                    if prev_mousex is None or prev_mousey is None:
                        prev_mousex = cur_mousex
                        prev_mousey = cur_mousey

                    mousedx = cur_mousex - prev_mousex
                    mousedy = cur_mousey - prev_mousey

                    prev_mousex = cur_mousex
                    prev_mousey = cur_mousey
                    
                    curr_keys_pressed = stateManager.keys_pressed[:]

                    prev_input.clear()
                    prev_input.extend(curr_keys_pressed)
                    prev_input.extend([mousedx, mousedy])

                    #this_frame_key_count = stateManager.update_keys_frame_count()
                    if len(input_history) > dataEncoder.HISTORY_LENGTH:
                        input_history.pop(0)
                        input_history.append(prev_input)
                    
                    
                    if stateManager.is_recording:
                        #img = cv2.Canny(img, threshold1=0, threshold2=200)
                        flat = img.reshape([resolution, 1])
                        
                        self.write_state_to_output(f"sample{sampleNum}", flat,\
                            mousedx, mousedy, curr_keys_pressed, input_history)
                        sampleNum +=1
                    # time when we finish processing for this frame
                    new_frame_time = time.time()

                    # fps will be number of frame processed in given time frame
                    # since their will be most of time error of 0.001 second
                    # we will be subtracting it to get more accurate result
                    fps = 1/(new_frame_time-prev_frame_time)
                    prev_frame_time = new_frame_time

                    # converting the fps into integer
                    fps = int(fps)

                    # converting the fps to string so that we can display it on frame
                    # by using putText function
                    key = "key: None"
                    debug_keys = stateManager.get_keys_pressed()
                    if len(debug_keys) > 0:
                        key = f"key: {str(debug_keys)}"
                    mouse = f"mouse: ({mousedx}, {mousedy})"

                    fps = f"fps: {fps}"

                    # putting the FPS count on the frame
                    cv2.putText(img, fps, (7, 25), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.putText(img, f"Recording: {stateManager.is_recording}", (70, 25), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.putText(img, key, (7, 55), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.putText(img, mouse, (7, 85), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

                    cv2.imshow('collector', np.array(img))
                    if cv2.waitKey(25) & 0xFF == ord('p'):
                        cv2.destroyAllWindows()
                        stateManager.is_not_exiting = False
                        break
            mouse_listener.join()
        key_listener.join()
                        