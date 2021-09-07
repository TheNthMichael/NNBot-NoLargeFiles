import cv2
import pickle
import dataEncoder
import numpy as np
import stateManager
import time


the_file = open("TestData/dataset_new_method.pickle", "rb")

training_inputs_frames = []
training_inputs_histories = []

training_outputs = []

input_history = [dataEncoder.BLANK_CLASS_OUTPUT[:] for _ in range(dataEncoder.HISTORY_LENGTH)]

chunk_size = 10000000 # Good luck collecting more samples than this
print(f"Getting dataset with chunksize: {chunk_size}")
try:
    for i in range(chunk_size):
        sample = pickle.load(the_file)
        mouse = sample.raw_mouse
        keys = sample.inputs
        frame = sample.frame
        mouse_encoded = dataEncoder.mouse_to_classification(mouse)
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


for i in range(len(training_inputs_frames)):
    img = training_inputs_frames[i]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (1920, 1080))
    inputshistory = training_inputs_histories[i]

    i = 0

    for input in inputshistory:
        keys, mousex, mousey = dataEncoder.output_to_mappings(input)
        cv2.putText(img, f"history: {keys}, {mousex}, {mousey}", (7, 25 + (i * 10)), cv2.FONT_HERSHEY_COMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
        i += 1
        
    
    time.sleep(1/12)

    
    cv2.imshow('player', img)

    if cv2.waitKey(25) & 0xFF == ord('p'):
        cv2.destroyAllWindows()
        stateManager.is_not_exiting = False
        break