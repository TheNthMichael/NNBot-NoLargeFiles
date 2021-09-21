import os
import math
import pickle
import numpy as np
import stateManager
import keybindHandler

class DataLoader:
    """Creates a new instance of the data loader.
    
    normalize is a flag which when set will normalize all data points to a range 0, 1.
    some data points such as mouse dp's require normalization so that they can be
    un-normalized afterward to retrieve a normal mouse value as all values get squished
    into the (0, 1) range of the sigmoid function regardless. The reason to normalize all
    values is to ensure one value doesn't oversaturate others."""
    def __init__(self, path, normalize=False, chunk_size: int=None) -> None:
        self.path = path
        self.the_file = open(self.path, 'rb')
        self.normalize = normalize
        self.chunk_size = chunk_size

        self.look_back_time = stateManager.HISTORY_LENGTH

        self.consumed_all_samples = False
        

    """
    format of data should mirror the following:
    X2 = [(np.random.rand(1920 // 8, 1080 // 8, 3) * 255)//1 for i in range(100)]
    X2 = np.asarray(X2, dtype=np.float32)
    X2 = X2 / 255 # Normlization of image data

    X3 = [[[randint(0, 1) for w in range(20)] for j in range(5)] for i in range(100)]
    X3 = np.asarray(X3)

    y = [[randint(0, 1) for w in range(20)] for i in range(100)]"""
    def get_training_set(self, chunk_size: int=None):
        assert(chunk_size is None or chunk_size > 0)

        training_inputs_frames = []
        training_inputs_histories = []

        training_outputs = []

        # Generate an array of empty histories
        output_template = [1 for _ in keybindHandler.ACTION_TO_BUTTON]
        output_template.extend(keybindHandler.mouse_to_classification([300,300]))
        input_history = [output_template[:] for _ in range(stateManager.HISTORY_LENGTH)]

        if not chunk_size:
            chunk_size =  stateManager.MAX_CHUNK_SIZE
        print(f"Getting dataset with chunksize: {chunk_size}")
        try:
            for _ in range(chunk_size):
                sample = pickle.load(self.the_file)
                mouse = sample.raw_mouse
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
        except Exception:
            print("Consumed all samples")
            self.consumed_all_samples = True
        return [training_inputs_frames[:], training_inputs_histories[:], training_outputs[:]]
        
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if not self.consumed_all_samples:
            return self.get_training_set(self.chunk_size)
        raise StopIteration