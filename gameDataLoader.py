import os
import numpy as np

def getSizes(path :str):
    for filename in os.listdir(path):
        filename = os.path.join(path, filename)
        print(f"Opened {filename}.")
        file = np.load(filename)
        keys = file['keys']
        mouse = file['mouse']
        input = file['image']
        input_size = input.shape[0]
        output = np.append(keys, mouse)
        output = np.transpose(output)
        print(f"SHAPE: {output.shape}\t{input.shape}")
        output_size = output.shape[0]
        return input_size, output_size

def get_loader(path :str, batchSize :int):
    def loader():
        trainingInputs = []
        trainingOutputs = []
        encoder = DataEncoderAndDecoder()
        encoder.find_mouse_range(path)
        input_size = 0
        output_size = 0
        batchCount = 0
        for filename in os.listdir(path):
            filename = os.path.join(path, filename)
            file = np.load(filename)
            keys = file['keys']
            mouse = file['mouse']
            mouse[0], mouse[1] = encoder.map_to_sigmoid(mouse[0], mouse[1])
            input = file['image']
            input_size = input.shape[0]
            output = np.append(keys, mouse)
            output = output.reshape([output.shape[0], 1])
            output_size = output.shape[0]
            trainingInputs.append(input)
            trainingOutputs.append(output)
            if getSizes:
                yield input_size, output_size
            if batchCount > batchSize:
                batchCount = 0
                data_set = zip(trainingInputs[:], trainingOutputs[:])
                trainingInputs.clear()
                trainingOutputs.clear()
                yield data_set, input_size, output_size
        if len(trainingInputs) > 0:
            data_set = zip(trainingInputs[:], trainingOutputs[:])
            trainingInputs.clear()
            trainingOutputs.clear()
            yield data_set, input_size, output_size
    return loader

"""The loader for image and input data for the model.
Format for input and output data
input: [0.5, 0.1, ..., 0.9] Range = (0, 1) Image Data
output: [Keys=(0.23, 0.99, 0.12, 0.98, 0.1, 0.76, 0.99), Mouse=(0.55, 0.3)] Range = (0, 1) Keyboard discrete states and Mouse real valued states.
"""
def loader(path :str, batchSize :int):
    trainingInputs = []
    trainingOutputs = []
    encoder = DataEncoderAndDecoder()
    encoder.find_mouse_range(path)
    input_size = 0
    output_size = 0
    batchCount = 0
    for filename in os.listdir(path):
        filename = os.path.join(path, filename)
        file = np.load(filename)
        keys = file['keys']
        mouse = file['mouse']
        mouse[0], mouse[1] = encoder.map_to_sigmoid(mouse[0], mouse[1])
        input = file['image']
        input_size = input.shape[0]
        output = np.append(keys, mouse)
        output = output.reshape([output.shape[0], 1])
        output_size = output.shape[0]
        trainingInputs.append(input)
        trainingOutputs.append(output)
        if batchCount > batchSize:
            batchCount = 0
            data_set = zip(trainingInputs[:], trainingOutputs[:])
            trainingInputs.clear()
            trainingOutputs.clear()
            yield data_set, input_size, output_size
    if len(trainingInputs) > 0:
        data_set = zip(trainingInputs[:], trainingOutputs[:])
        trainingInputs.clear()
        trainingOutputs.clear()
        yield data_set, input_size, output_size
    
        
