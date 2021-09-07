from os import stat
import nnmodel
import cv2
import numpy as np
from gameDataLoader import *
from datetime import datetime
import ctypes
import time
import sys
from mss import mss
from PIL import Image
from pynput.keyboard import Key, Controller as KeyController
from pynput.mouse import Button, Controller as MouseController

dirname = os.path.dirname(__file__)

pressed_keys = []

def train(batch_size, eta):
    #training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    # For getting the sizes, we will read the first sample, then ignore it.
    myData = os.path.join(dirname, "TestData", "dataset_tps_akina10k")
    inputsize, outputsize = getSizes(myData)
    print(f"InputSize: {inputsize}\nOutputSize: {outputsize}")
    net = nnmodel.Network([inputsize, 40, 10, 40, outputsize])
    for data_set, inputsize, outputsize in loader(myData, batch_size):
        print("Starting new batch")
        net.SGD(data_set, 10, batch_size, eta, test_data=data_set)
    
    nnmodel.save_model(f"SavedModel{datetime.now().strftime('%Y%m%d%H%M%S')}", net)

def on_press(key):
    pressed_keys.append(str(key).lower())
    try:
        print('Bot pressed key {0}.'.format(key.char))
          
    except AttributeError:
        print('Bot pressed special key {0}.'.format(key))
        return 
           
stateManager = InputStateManager()
               
def on_release(key):
    pressed_keys.remove(str(key).lower())
    if key == Key.f2:
        stateManager.isRecording = not stateManager.isRecording
    if key == Key.esc:
        stateManager.isNotExiting = not stateManager.isNotExiting
    print('Bot released key {0}.'.format(key))
    return stateManager.isNotExiting

def play(model :nnmodel.Network):
    myData = os.path.join(dirname, "TestData", "dataset_tps_akina10k")
    stateManager.encoder.find_mouse_range(myData)
    keyboard = KeyController()
    mouse = MouseController()
    # used to record the time when we processed last frame
    prev_frame_time = 0

    # used to record the time at which we processed current frame
    new_frame_time = 0
    mon = {'top': 0, 'left': 0, 'width': 1920, 'height': 1080}
    sct = mss()
    with KeyListener(on_press = on_press,
              on_release = on_release) as key_listener:
        while stateManager.isNotExiting:
            sct.get_pixels(mon)
            img = Image.frombytes('RGB', (sct.width, sct.height), sct.image)
            img = np.array(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            small = cv2.resize(img, (1920 //  stateManager.scalar, 1080 //  stateManager.scalar))
            img = small
            resolution = (1920 // stateManager.scalar) * (1080 //  stateManager.scalar)

            #img = cv2.Canny(img, threshold1=0, threshold2=200)
            if stateManager.isRecording:
                flat = small.reshape([resolution, 1])
                output = model.feedforward(flat)
                mouseOutput = output[-2:]
                keys = output[:-2]
                keys = [1 if x > 0.6 else 0 for x in keys]
                
                printkey = f"Keys: {str(output)}"

                #mouseOutput[0], mouseOutput[1] = stateManager.encoder.map_to_real(mouseOutput[0], mouseOutput[1])
                mouseOutput = [(x - 0.9) * 10 for x in mouseOutput]
                printmouse = f"Mouse: ({int(mouseOutput[0])}, {int(mouseOutput[1])})"
                for i in range(len(keys)):
                    key = stateManager.encoder.keyDecoder[i]
                    if keys[i] == 1:
                        keyboard.press(key)
                    else:
                        keyboard.release(key)
                print(f"Moving Mouse: ({int(mouseOutput[0])}, {int(mouseOutput[1])})")
                x = int(mouseOutput[0])
                y = int(mouseOutput[1])
                ctypes.windll.user32.mouse_event(0x01, x, y, 0, 0)

                # converting the fps to string so that we can display it on frame
                # by using putText function

                # time when we finish processing for this frame
                new_frame_time = time.time()

                # fps will be number of frame processed in given time frame
                # since their will be most of time error of 0.001 second
                # we will be subtracting it to get more accurate result
                fps = 1/(new_frame_time-prev_frame_time)
                prev_frame_time = new_frame_time

                # converting the fps into integer
                fps = int(fps)

                fps = f"fps: {fps}"

                # putting the FPS count on the frame
                if "--debug" in sys.argv:
                    cv2.putText(img, fps, (7, 25), cv2.FONT_HERSHEY_COMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.putText(img, printkey, (7, 55), cv2.FONT_HERSHEY_COMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.putText(img, printmouse, (7, 85), cv2.FONT_HERSHEY_COMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
            else:
                for pkey in pressed_keys:
                    keyboard.release(pkey)

            cv2.imshow('display', img)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                stateManager.isNotExiting = False
                break
        key_listener.join()

    pass

def load_and_play(path :str) -> nnmodel.Network:
    model = nnmodel.load_model(path)
    play(model)
    

if __name__ == "__main__":
    #train(9408, 0.05)
    load_and_play('model_tps_akina10k')