import numpy as np
import dataEncoder
import math
import cv2
import time
import os
import pickle
import keybindHandler
import stateManager
from datetime import datetime
from pynput.keyboard import Key, Listener as KeyListener
from pynput.mouse import Listener as MouseListener
from frameHandler import *

def on_press_handler(key):
    keybindHandler.record_input(key, pressed=True)
    try:
        print('alphanumeric key {0} pressed'.format(key.char))
          
    except AttributeError:
        print('special key {0} pressed'.format(key))

def on_release_handler(key):
    if key == Key.f3:
        stateManager.is_recording = not stateManager.is_recording
        return stateManager.is_not_exiting
    if key == Key.esc:
        stateManager.is_not_exiting = not stateManager.is_not_exiting
        return stateManager.is_not_exiting 
    keybindHandler.record_input(key, pressed=False)
    print('{0} released'.format(key))
    return stateManager.is_not_exiting

def on_move_handler(x, y):
    keybindHandler.last_mouse_moved_x = x
    keybindHandler.last_mouse_moved_y = y
    #print('Pointer moved to {0}'.format((x, y)))
    return stateManager.is_not_exiting

def on_click_handler(x, y, button, pressed):
    keybindHandler.record_input(button, pressed=pressed)
    print(f'{str(button)} is pressed {pressed} at {(x,y)}')
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
        self.frameHandler = FrameHandler(stateManager.monitor_region, stateManager.FPS)
    
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
        prev_time = 0

        cv2.namedWindow('collector')
        try:
            with KeyListener(on_press = on_press_handler,
                on_release = on_release_handler) as key_listener:
                with MouseListener(on_move = on_move_handler, on_click = on_click_handler) as mouse_listener:
                    while stateManager.is_not_exiting:
                        self.frameHandler.update()
                        img = self.frameHandler.get_current_frame()
                        img = cv2.resize(img, stateManager.screen_cap_sizes)

                        cur_time = time.time()
                        cur_mousex = keybindHandler.last_mouse_moved_x
                        cur_mousey = keybindHandler.last_mouse_moved_y

                        if prev_mousex is None or prev_mousey is None:
                            prev_mousex = cur_mousex
                            prev_mousey = cur_mousey

                        mousedx = cur_mousex - prev_mousex
                        mousedy = cur_mousey - prev_mousey

                        #mousedx *= 1 / (cur_time - prev_time)
                        #mousedy *= 1 / (cur_time - prev_time)

                        #mousedx = math.ceil(mousedx)
                        #mousedy = math.ceil(mousedy)

                        prev_mousex = cur_mousex
                        prev_mousey = cur_mousey
                        
                        curr_keys_pressed = keybindHandler.buttons_pressed_by_human
                        
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
                        debug_keys = keybindHandler.get_keys_pressed()
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
                        if cv2.waitKey(1) & 0xFF == ord('p'):
                            cv2.destroyAllWindows()
                            stateManager.is_not_exiting = False
                            break

                mouse_listener.join()
            key_listener.join()
        except Exception as e:
            print(e)
        finally:
            self.dataset_file.close()
            cv2.destroyAllWindows()
            stateManager.is_not_exiting = False
            print("DataCollector Closing...")