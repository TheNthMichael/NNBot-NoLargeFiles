from re import A
import threading
import keras
import cv2
import traceback
import time
import numpy as np
from pynput.keyboard import Key, Controller as KeyController, Listener as KeyListener
from pynput.mouse import Listener as MouseListener, Controller as MouseController
import stateManager
import actionHandler
import keybindHandler
from frameHandler import *
import keras.backend as K
import random
import tensorflow as tf

"""Handle mouse button presses and releases."""
def on_click_handler(x, y, button, pressed):
    print(f'{str(button)} is pressed {pressed} at {(x,y)}')
    return not stateManager.is_exiting.is_set()

"""Handle keyboard button presses."""
def on_press_handler(key):
    return not stateManager.is_exiting.is_set()

"""Handle keyboard button releases.
Handles the user input for state management. (toggle recording, set exit event)"""
def on_release_handler(key):
    if key == Key.f6:
        stateManager.toggle_recording()
        not stateManager.is_exiting.is_set()
    if key == Key.f9:
        stateManager.is_exiting.set()
        return not stateManager.is_exiting.is_set()
    return not stateManager.is_exiting.is_set()

"""Random loss function not used due to classes not needing to be weighted separately."""
def my_loss(targets, logits):
    weights = np.array([0.9 for _ in range(len(keybindHandler.EMPTY_CLASSES_ONEHOT))])
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

"""Start the model and wait for user input to toggle 'recording' event which is used to toggle the bot here."""
def play(model_path: str):
    
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    model = keras.models.load_model(model_path, custom_objects={"f1": f1, "my_loss": my_loss})
    frame_handler = FrameHandler(stateManager.monitor_region, stateManager.FPS)
    action_handler = actionHandler.ActionHandler(500)
    #data_lock = threading.Lock()
    action_thread = threading.Thread(target=actionHandler.action_handler_thread, args=(action_handler, stateManager.is_exiting, stateManager.is_recording))
    try:
        # used to record the time when we processed last frame
        prev_frame_time = 0

        cv2.namedWindow('player')

        # Generate an array of empty histories
        output_template = [1 for _ in keybindHandler.ACTION_TO_BUTTON]
        output_template.extend(keybindHandler.mouse_to_classification([300,300]))
        input_history = [output_template[:] for _ in range(stateManager.HISTORY_LENGTH)]

        frame_handler.update()
        img = frame_handler.get_current_frame()
        img = cv2.resize(img, stateManager.screen_cap_sizes)
        frame_template = img * 0
        frames_history = [frame_template[:] for _ in range(stateManager.HISTORY_LENGTH)]

        # used to record the time at which we processed current frame
        new_frame_time = 0
        with KeyListener(on_press = on_press_handler,
                on_release = on_release_handler) as key_listener:
            action_thread.start()
            while not stateManager.is_exiting.is_set():
                frame_handler.update()
                img = frame_handler.get_current_frame()
                img = cv2.resize(img, stateManager.screen_cap_sizes)
                
                if stateManager.is_recording.is_set():
                    img = img / 255
                    frames_history.pop(0)
                    frames_history.append(img)
                    X2 = [frames_history for i in range(1)]
                    X2 = np.asarray(X2)
                    X3 = [input_history for i in range(1)]
                    X3 = np.asarray(X3)

                    #output = model.predict([X2, X3])[0]
                    output = model([X2, X3])[0]

                    #print(output)

                    keys_output, mousex_output, mousey_output = keybindHandler.output_to_mappings(output)
                    keys = [1 if x > stateManager.KEY_THRESHOLD else 0 for x in keys_output]

                    mousex_ind = np.argmax(mousex_output)
                    mousey_ind = np.argmax(mousey_output)

                    input_history.pop(0)
                    keys_h = keys[:]
                    mousex_h = [0 for _ in keybindHandler.MOUSE_CLASSES]
                    mousex_h[mousex_ind] = 1
                    mousey_h = [0 for _ in keybindHandler.MOUSE_CLASSES]
                    mousey_h[mousey_ind] = 1
                    keys_h.extend(mousex_h)
                    keys_h.extend(mousey_h)
                    input_history.append(keys_h)

                    mousex = keybindHandler.MOUSE_CLASSES[mousex_ind]
                    mousey = keybindHandler.MOUSE_CLASSES[mousey_ind]

                    # Update the keys and mouse inputs - May need a data lock here to avoid issues with concurrency.
                    action_handler.set_controller_action(keys, mousex * 100, mousey * 100)
                    #for i in range(stateManager.FPS):
                        #action_handler.update()    


                    printmouse = f"Mouse: ({int(mousex)}, {int(mousey)})"

                    print(printmouse)
                    

                    printkey = f"Keys: {str(keybindHandler.get_your_keys_pressed(keys))}"

                    # converting the fps to string so that we can display it on frame
                    # by using putText function

                    # time when we finish processing for this frame
                    new_frame_time = time.time()

                    # fps will be number of frame processed in given time frame
                    # since their will be most of time error of 0.001 second
                    # we will be subtracting it to get more accurate result
                    fps = 1/(max(new_frame_time-prev_frame_time, 0.000000001))
                    prev_frame_time = new_frame_time

                    # converting the fps into integer
                    fps = int(fps)

                    fps = f"fps: {fps}"

                    # putting the FPS count on the frame
                    cv2.putText(img, fps, (7, 25), cv2.FONT_HERSHEY_COMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.putText(img, printkey, (7, 55), cv2.FONT_HERSHEY_COMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.putText(img, printmouse, (7, 85), cv2.FONT_HERSHEY_COMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
                
                else:
                    # time when we finish processing for this frame
                    new_frame_time = time.time()

                    # fps will be number of frame processed in given time frame
                    # since their will be most of time error of 0.001 second
                    # we will be subtracting it to get more accurate result
                    fps = 1/(max(new_frame_time-prev_frame_time, 0.000000001))
                    prev_frame_time = new_frame_time
                    fps = int(fps)
                    fps = f"fps: {fps}"
                    cv2.putText(img, fps, (7, 25), cv2.FONT_HERSHEY_COMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)

                cv2.imshow('player', img)

                if cv2.waitKey(5) & 0xFF == ord('p'):
                    cv2.destroyAllWindows()
                    stateManager.is_exiting.set()
                    break
            key_listener.join()
    except Exception as e:
        print(f"Error in BotPlayer: {e}")
        traceback.print_exc()
    finally:
        cv2.destroyAllWindows()
        action_handler.release_pressed_buttons()
        action_thread.join()
        stateManager.is_exiting.set()