from os import stat
from actionHandler import action_handler_thread
import keybindHandler
import cv2
import pickle
import numpy as np
import stateManager
import time
import keras
import keras.backend as K
import random
import tensorflow as tf

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


the_file = open("TestData/aottg_overlay_test.pickle", "rb")

the_model = "Models/km_aottg_ov.model"

model = keras.models.load_model(the_model, custom_objects={"f1": f1, "my_loss": my_loss})

# Generate an array of empty histories
output_template = [1 for _ in keybindHandler.ACTION_TO_BUTTON]
output_template.extend(keybindHandler.mouse_to_classification([300,300]))
input_history_bot = [output_template[:] for _ in range(stateManager.HISTORY_LENGTH)]

training_inputs_frames = []
training_inputs_histories = []

training_outputs = []

input_history = [keybindHandler.EMPTY_CLASSES_ONEHOT[:] for _ in range(stateManager.HISTORY_LENGTH)]

real_mouse = []

chunk_size = stateManager.MAX_CHUNK_SIZE # Good luck collecting more samples than this
print(f"Getting dataset with chunksize: {chunk_size}")
try:
    for i in range(chunk_size):
        sample = pickle.load(the_file)
        mouse = sample.raw_mouse
        real_mouse.append(mouse)
        keys = sample.inputs
        frame = sample.frame
        mouse_encoded = keybindHandler.mouse_to_classification(mouse)
        input_history_sample = np.append(keys, mouse_encoded)
        input_history.pop(0)
        input_history.append(input_history_sample[:])
        training_inputs_frames.append(frame[:])
        training_inputs_histories.append(input_history[:])

        output = np.append(keys, mouse_encoded)
        training_outputs.append(output[:])
except Exception as e:
    print(e)
trdata = [training_inputs_frames[:], training_inputs_histories[:], training_outputs[:]]

num_correct = 0
num_total = 1

for i in range(len(training_inputs_frames)):
    frame = training_inputs_frames[i]

    # Bot
    X2 = [(frame / 255) for i in range(1)]
    X2 = np.asarray(X2)
    X3 = [input_history_bot for i in range(1)]
    X3 = np.asarray(X3)

    output = model([X2, X3])[0]

    #print(output)

    keys_output, mousex_output, mousey_output = keybindHandler.output_to_mappings(output)
    #keys_bot = [1 if x >= dataEncoder.KEY_THRESHOLD else 0 for x in keys_output]
    keys_bot = [1 if x > stateManager.KEY_THRESHOLD else 0 for x in keys_output]

    mousex_ind = np.argmax(mousex_output)
    mousey_ind = np.argmax(mousey_output)

    input_history_bot.pop(0)
    keys_h = keys_bot[:]
    mousex_h = [0 for _ in keybindHandler.MOUSE_CLASSES]
    mousex_h[mousex_ind] = 1
    mousey_h = [0 for _ in keybindHandler.MOUSE_CLASSES]
    mousey_h[mousey_ind] = 1
    bot_output = np.append(keys_h, mousex_h)
    bot_output = np.append(bot_output, mousey_h)
    input_history_bot.append(bot_output)

    mousex = keybindHandler.MOUSE_CLASSES[mousex_ind]
    mousey = keybindHandler.MOUSE_CLASSES[mousey_ind]


    #printmouse = f"Mouse: ({int(mousex)}, {int(mousey)})"
    # End Bot
    frame = cv2.resize(frame, (1920, 1080))
    inputshistory = training_inputs_histories[i]

    j = 0
    cv2.putText(frame, f"Human", (7, 25 + (j * 10)), cv2.FONT_HERSHEY_COMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
    j += 1
    cv2.putText(frame, f"history[map]: {keybindHandler.print_buttons()}, {keybindHandler.MOUSE_CLASSES}, {keybindHandler.MOUSE_CLASSES}", (7, 25 + (j * 10)), cv2.FONT_HERSHEY_COMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
    j += 1
    for input in range(len(inputshistory)):
        keys, mousex, mousey = keybindHandler.output_to_mappings(inputshistory[input])
        cv2.putText(frame, f"history[t-{stateManager.HISTORY_LENGTH - input}]: {keys}, {mousex}, {mousey} | Mouse: ({keybindHandler.MOUSE_CLASSES[np.argmax(mousex)]}, {keybindHandler.MOUSE_CLASSES[np.argmax(mousey)]}) | Real Mouse: ({real_mouse[i][0]}, {real_mouse[i][1]})", (7, 25 + (j * 10)), cv2.FONT_HERSHEY_COMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
        j += 1

    j += 1
    cv2.putText(frame, f"Bot", (7, 25 + (j * 10)), cv2.FONT_HERSHEY_COMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
    j += 1
    cv2.putText(frame, f"history[map]: {keybindHandler.print_buttons()}, {keybindHandler.MOUSE_CLASSES}, {keybindHandler.MOUSE_CLASSES}", (7, 25 + (j * 10)), cv2.FONT_HERSHEY_COMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
    j += 1
    for input in range(len(input_history_bot)):
        keys, mousex, mousey = keybindHandler.output_to_mappings(input_history_bot[input])
        cv2.putText(frame, f"history[t-{stateManager.HISTORY_LENGTH - input}]: {keys}, {mousex}, {mousey} | Mouse: ({keybindHandler.MOUSE_CLASSES[np.argmax(mousex)]}, {keybindHandler.MOUSE_CLASSES[np.argmax(mousey)]})", (7, 25 + (j * 10)), cv2.FONT_HERSHEY_COMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
        j += 1
    
    for input in range(len(input_history_bot)):
        keys, mousex, mousey = keybindHandler.output_to_mappings(inputshistory[input])
        keys_b, mousex_b, mousey_b = keybindHandler.output_to_mappings(input_history_bot[input])
        human = inputshistory[input]
        bot = input_history_bot[input]
        for i in range(len(human)):
            if human[i] == 1:
                if bot[i] == human[i]:
                    num_correct += 1
                num_total += 1
    cv2.putText(frame, f"Accuracy: {num_correct} / {num_total} ({int(num_correct * 100 / num_total)} %)", (7, 25 + (j * 10)), cv2.FONT_HERSHEY_COMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
    j += 1

        
    
    #time.sleep(1/12)

    
    cv2.imshow('player', frame)

    if cv2.waitKey(25) & 0xFF == ord('p'):
        cv2.destroyAllWindows()
        stateManager.is_exiting.set()
        break
print(f"Total Accuracy: {num_correct} / {num_total} ({int(num_correct * 100 / num_total)} %)")