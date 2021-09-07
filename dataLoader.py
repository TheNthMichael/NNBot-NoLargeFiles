import os
import math
import pickle
import numpy as np
import dataEncoder
import stateManager
import cv2

class DataLoader:
    class NormalizedAttributes:
        def __init__(self) -> None:
            self.xMouseMin = float('inf')
            self.xMouseMax = float('-inf')
            self.yMouseMin = float('inf')
            self.yMouseMax = float('-inf')
            self.imageValMin = 0
            self.imageValMax = 1

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
        self.map_mouse_to_sigmoid = None

        # Our model will hold data for approx 12 frames of a key being held down.
        # after this point, it will just be flagged as "key has been down as long
        # as I've known."
        self.look_back_time = dataEncoder.HISTORY_LENGTH
        # As x approaches infinity norm_map approaches 1.
        self.zero_map = np.vectorize(lambda x: 1 - math.exp(-x/self.look_back_time))
        # As x approaches infinity invnorm_map approaches 0.
        self.one_map = np.vectorize(lambda x: math.exp(-x/self.look_back_time))

        self.vec_linmap = np.vectorize(dataEncoder.linmap)

        self.forget_curve = np.vectorize(lambda c, t: (1 - c) * self.vec_linmap(self.zero_map(t), 0, 1, 0, 0.5) + c * self.vec_linmap(self.one_map(t), 1, 0, 1, 0.5))
        self.consumed_all_samples = False
        if self.normalize:
            self.get_normalize_attributes()


    def get_normalize_attributes(self):
        self._norm_attrs = self.NormalizedAttributes()
        self._norm_attrs.imageValMin = 0
        self._norm_attrs.imageValMax = 255

        self.map_mouse_to_sigmoid = lambda x, y: (dataEncoder.linmap(x,\
           self._norm_attrs.xMouseMin, self._norm_attrs.xMouseMax, 0, 1),\
               dataEncoder.linmap(y, self._norm_attrs.yMouseMin, self._norm_attrs.yMouseMax, 0, 1))
        

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

        input_history = [dataEncoder.BLANK_CLASS_OUTPUT[:] for _ in range(dataEncoder.HISTORY_LENGTH)]

        if not chunk_size:
            chunk_size = 10000000 # Good luck collecting more samples than this
        print(f"Getting dataset with chunksize: {chunk_size}")
        try:
            for i in range(chunk_size):
                sample = pickle.load(self.the_file)
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