import numpy as np
import sys
import cv2
import time
import os
from datetime import datetime
from mss import mss
from PIL import Image
import pynput
from pynput.keyboard import Key, Listener as KeyListener
from pynput.mouse import Listener as MouseListener
from dataEncoding import *

dirname = os.path.dirname(__file__)
testdir = os.path.join(dirname, "TestData")
thistestdir = os.path.join(testdir, f"dataset{datetime.now().strftime('%Y%m%d%H%M%S')}")
os.mkdir(thistestdir)


keys = []

stateManager = InputStateManager()
   
def on_press(key):
    if not str(key).lower() in keys:
        keys.append(str(key).lower())
    try:
        print('alphanumeric key {0} pressed'.format(key.char))
          
    except AttributeError:
        print('special key {0} pressed'.format(key))
           

               
def on_release(key):
    global recordingOn
    if key == Key.f2:
        stateManager.isRecording = not stateManager.isRecording
        return stateManager.isNotExiting
    if str(key).lower() in keys:
        keys.remove(str(key).lower())
    print('{0} released'.format(key))
    return stateManager.isNotExiting

def on_move(x, y):
    stateManager.on_move(x,y)
    print('Pointer moved to {0}'.format(
        (x, y)))
    return stateManager.isNotExiting


mon = {'top': 0, 'left': 0, 'width': 1920, 'height': 1080}

sct = mss()


sampleNum = 0

# used to record the time when we processed last frame
prev_frame_time = 0

# used to record the time at which we processed current frame
new_frame_time = 0
with KeyListener(on_press = on_press,
              on_release = on_release) as key_listener:
    with MouseListener(on_move = on_move) as mouse_listener:
        while True:
            sct.get_pixels(mon)
            img = Image.frombytes('RGB', (sct.width, sct.height), sct.image)
            img = np.array(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = cv2.resize(img, (1920 //  stateManager.scalar, 1080 //  stateManager.scalar))
            resolution = (1920 // stateManager.scalar) * (1080 //  stateManager.scalar)

            #img = cv2.Canny(img, threshold1=0, threshold2=200)
            flat = img.reshape([resolution, 1])
            file = os.path.join(thistestdir, f"sample{sampleNum}")
            if stateManager.isRecording:
                sampleNum +=1
                stateManager.write_state_to_output(file, flat, keys)
            
            #img = cv2.resize(img, (1920 // 8, 1080 // 8))

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
            if len(keys) > 0:
                key = f"key: {str(keys)}"
            mouse = f"mouse: ({stateManager.lastDx}, {stateManager.lastDy})"

            fps = f"fps: {fps}"

            # putting the FPS count on the frame
            if "--debug" in sys.argv:
                cv2.putText(img, fps, (7, 25), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(img, f"Recording: {stateManager.isRecording}", (70, 25), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(img, key, (7, 55), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(img, mouse, (7, 85), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow('test', np.array(img))
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                stateManager.isNotExiting = False
                break
        mouse_listener.join()
    key_listener.join()