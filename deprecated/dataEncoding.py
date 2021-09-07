import os
import numpy as np
import pynput
from pynput.keyboard import Key, Listener as KeyListener
from pynput.mouse import Listener as MouseListener

"""Transforms the value x from the input range to the output range."""
def linmap(x, in_min, in_max, out_min, out_max):
        return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

"""Clamps the value x to the minv and maxv range."""
def clamp(x, minv, maxv):
    assert(minv <= maxv)
    return min(max(x, minv), maxv)

"""We need to map the keys to indices of an array for the classifier
We also need to map the mouse movements max and min to 0-1 for the classifier
We will need to undo this transformation afterwards."""
class DataEncoderAndDecoder:
    def __init__(self) -> None:
        self.keyEncoder = {
            "'w'": 0,
            "'a'": 1,
            "'s'": 2,
            "'d'": 3,
            "'q'": 4,
            "'e'": 5,
            "key.shift": 6,
        }

        self.keyDecoder = {
            0 : 'w',
            1 : 'a',
            2 : 's',
            3 : 'd',
            4 : 'q',
            5 : 'e',
            6 : Key.shift,
        }

        # We need to map mouse dp's to a range (0, 1) for the classifier.
        # The result will need to be remapped to this original range. Due
        # to this issue, training can only be done on one set of data before
        # becoming untrainable otherwise if we get larger min and max ranges
        # We will break our model or require clamping which doesn't reflect the real data.
        self.xMouseMin = 0
        self.xMouseMax = 0

        self.yMouseMin = 0
        self.yMouseMax = 0
    
    def find_mouse_range(self, data_folder):
        for filename in os.listdir(data_folder):
            filename = os.path.join(data_folder, filename)
            file = np.load(filename)
            mouse = file['mouse']
            self.xMouseMin = min(self.xMouseMin, mouse[0])
            self.xMouseMax = max(self.xMouseMax, mouse[0])
            self.yMouseMin = min(self.yMouseMin, mouse[1])
            self.yMouseMax = max(self.yMouseMax, mouse[1])
        xAbsMax = max(abs(self.xMouseMin), abs(self.xMouseMax))
        yAbsMax = max(abs(self.yMouseMin), abs(self.yMouseMax))
        self.xMouseMin = -xAbsMax
        self.xMouseMax = xAbsMax

        self.yMouseMin = -yAbsMax
        self.yMouseMax = yAbsMax
        print(f"Mouse Bounds:\nX: ({self.xMouseMin}, {self.xMouseMax})\nY: ({self.yMouseMin}, {self.yMouseMax})")

    def map_to_sigmoid(self, x: float, y: float):
        x = clamp(x, self.xMouseMin, self.xMouseMax)
        y = clamp(y, self.yMouseMin, self.yMouseMax)
        return linmap(x, self.xMouseMin, self.xMouseMax, 0, 1), linmap(y, self.yMouseMin, self.yMouseMax, 0, 1)

    def map_to_real(self, x: float, y: float):
        x = clamp(x, 0, 1)
        y = clamp(y, 0, 1)
        return linmap(x, 0, 1, self.xMouseMin, self.xMouseMax), linmap(y, 0, 1, self.yMouseMin, self.yMouseMax)

"""class InputStateManager:
    def __init__(self) -> None:
        self.encoder = DataEncoderAndDecoder()
        self.keys = [0 for _ in self.encoder.keyEncoder]
        self.lastMouseX = 0
        self.lastMouseY = 0
        self.lastDx = 0
        self.lastDy = 0
        self.isRecording = False
        self.isNotExiting = True
        self.scalar = 8
    
    def write_state_to_output(self, path, image, keys):

        # construct a list of enuemerated keypresses where its 1 if on, 0 otherwise.
        self.keys = [0 for _ in self.encoder.keyEncoder]
        for key in keys:
            if str(key).lower() in self.encoder.keyEncoder:
                self.keys[self.encoder.keyEncoder[str(key).lower()]] = 1
            else:
                print(f"Key: {str(key).lower()} is not in encoder.")
        np.savez(path, image=image, keys=np.array(self.keys), mouse=np.array([self.lastDx, self.lastDy]))
    
    def on_move(self, x, y):
        self.lastDx = x - self.lastMouseX
        self.lastDy = y - self.lastMouseY
        self.lastMouseX = x
        self.lastMouseY = y"""