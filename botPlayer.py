
import keras
import cv2
import ctypes
import time
import numpy as np
import dataEncoder
from mss import mss
from PIL import Image
from pynput.keyboard import Key, Controller as KeyController, Listener as KeyListener
from pynput.mouse import Listener as MouseListener
import stateManager
import keras.backend as K
import random

def on_press_handler(key):
    stateManager.try_add_key_pressed(key)
    try:
        pass
        #print('alphanumeric key {0} pressed'.format(key.char))
          
    except AttributeError:
        pass
        #print('special key {0} pressed'.format(key))
    return stateManager.is_not_exiting

def on_release_handler(key):
    if key == Key.f3:
        stateManager.is_recording = not stateManager.is_recording
        return stateManager.is_not_exiting
    stateManager.try_remove_key_pressed(key)
    #print('{0} released'.format(key))
    return stateManager.is_not_exiting

def my_loss(targets, logits):
    weights = np.array([0.9 for _ in range(len(dataEncoder.BLANK_CLASS_OUTPUT))])
    return K.sum(targets * -K.log(1 - logits + 1e-10) * weights + (1 - targets) * -K.log(1 - logits + 1e-10) * (1 - weights), axis=-1)

# This metric was taken from here: https://drive.google.com/file/d/1MOVhZhn0yv-Ngp0xK9jly-b_Ttx_2Tf7/view
# by the author of said paper who took it from a stackoverflow post listed there
def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall
    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (possible_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

def play(model_path: str):
    model = keras.models.load_model(model_path, custom_objects={"f1": f1, "my_loss": my_loss})
    keyboard = KeyController()
    sct = mss()
    # used to record the time when we processed last frame
    prev_frame_time = 0

    cv2.namedWindow('player')

    # Generate an array of empty histories
    output_template = [1 for _ in dataEncoder.KEY_TO_CODE_MAP]
    output_template.extend(dataEncoder.mouse_to_classification([300,300]))
    input_history = [output_template[:] for _ in range(dataEncoder.HISTORY_LENGTH)]

    # used to record the time at which we processed current frame
    new_frame_time = 0
    with KeyListener(on_press = on_press_handler,
            on_release = on_release_handler) as key_listener:
        while stateManager.is_not_exiting:
            sct.get_pixels(stateManager.monitor_region)
            img = Image.frombytes('RGB', (sct.width, sct.height), sct.image)
            img = np.array(img)
            img = cv2.resize(img, stateManager.screen_cap_sizes)
            
            if stateManager.is_recording:
                
                X2 = [(img / 255) for i in range(1)]
                X2 = np.asarray(X2)
                X3 = [input_history for i in range(1)]
                X3 = np.asarray(X3)

                output = model.predict([X2, X3])[0]

                print(output)

                keys_output, mousex_output, mousey_output = dataEncoder.output_to_mappings(output)
                keys = [1 if x > dataEncoder.KEY_THRESHOLD else 0 for x in keys_output]

                mousex_ind = np.argmax(mousex_output)
                mousey_ind = np.argmax(mousey_output)

                input_history.pop(0)
                keys_h = keys[:]
                mousex_h = [0 for _ in dataEncoder.MOUSE_CLASSES]
                mousex_h[mousex_ind] = 1
                mousey_h = [0 for _ in dataEncoder.MOUSE_CLASSES]
                mousey_h[mousey_ind] = 1
                keys_h.extend(mousex_h)
                keys_h.extend(mousey_h)
                input_history.append(keys_h)

                mousex = dataEncoder.MOUSE_CLASSES[mousex_ind]
                mousey = dataEncoder.MOUSE_CLASSES[mousey_ind]


                printmouse = f"Mouse: ({int(mousex)}, {int(mousey)})"
                for i in range(len(keys)):
                    key = dataEncoder.CODE_TO_KEY_MAP[i]
                    if keys[i] == 1 and stateManager.let_go_of_my_keys_please[i] == 0:
                        keyboard.press(key)
                    elif keys[i] == 0 and stateManager.let_go_of_my_keys_please[i] == 1:
                        keyboard.release(key)

                printkey = f"Keys: {str(stateManager.get_your_keys_pressed(keys))}"

                # Record keys pressed
                stateManager.update_pressed_keys(keys)

                #print(f"Moving Mouse: ({int(mousex)}, {int(mousey)})")
                x = int(mousex)
                y = int(mousey)
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
                cv2.putText(img, fps, (7, 25), cv2.FONT_HERSHEY_COMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(img, printkey, (7, 55), cv2.FONT_HERSHEY_COMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(img, printmouse, (7, 85), cv2.FONT_HERSHEY_COMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
            
            else:
                stateManager.let_go_of_pressed_keys(keyboard)

            cv2.imshow('player', img)

            if cv2.waitKey(25) & 0xFF == ord('p'):
                cv2.destroyAllWindows()
                stateManager.is_not_exiting = False
                break
        key_listener.join()