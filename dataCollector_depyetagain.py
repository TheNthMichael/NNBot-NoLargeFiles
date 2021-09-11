from genericpath import samefile
import numpy as np
import dataEncoder
import cv2
import time
import os
import pickle
import stateManager
from datetime import datetime
from mss import mss
from pil import Image
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

"""Format for each sample in our training set.

Required Preprocessing:
Normalize self.frame by dividing it by 255.
raw_mouse should be converted into a discrete set of possible mouse movements."""
class TrainingSample:
    def __init__(self, sample_name, frame, inputs, raw_mouse):
        self.sample_name = sample_name
        self.frame = frame
        self.inputs = inputs
        self.raw_mouse = raw_mouse

class DataCollector:
    def __init__(self, dataset_path:str=None) -> None:
        self.dataset_path = dataset_path
        if self.dataset_path == None:
            dirname = os.path.dirname(__file__)
            testdir = os.path.join(dirname, "TestData")
            if not os.path.exists(testdir):
                os.mkdir(testdir)
            
            self.dataset_path = os.path.join(testdir, \
                f"dataset{datetime.now().strftime('%Y%m%d%H%M%S')}.pickle")
        self.dataset_file = open(self.dataset_path, "wb")
        self.sct = mss()
    
    def write_state_to_output(self, training_sample):
        pickle.dump(training_sample, self.dataset_file)

    def run(self):
        # used to record the time when we processed last frame
        prev_frame_time = 0
        # used to record the time at which we processed current frame
        new_frame_time = 0
        sampleNum = 0
        prev_mousex = None
        prev_mousey = None

        cv2.namedWindow('collector')

        with KeyListener(on_press = on_press_handler,
              on_release = on_release_handler) as key_listener:
            with MouseListener(on_move = on_move_handler) as mouse_listener:
                while stateManager.is_not_exiting:
                    self.sct.get_pixels(stateManager.monitor_region)
                    img = Image.frombytes('RGB', (self.sct.width, self.sct.height), self.sct.image)
                    img = np.array(img)
                    #img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                    img = cv2.resize(img, stateManager.screen_cap_sizes)

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
                    
                    if stateManager.is_recording: 
                        self.write_state_to_output(TrainingSample(f"sample{sampleNum}", img, curr_keys_pressed, [mousedx, mousedy]))
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
        self.dataset_file.close()
        print("DataCollector Closing...")