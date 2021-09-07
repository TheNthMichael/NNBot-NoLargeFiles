import nnmodel
import mnist_loader
import camera_loader
import cv2
import numpy as np
from gameDataLoader import *

dirname = os.path(__file__)

def rnet(n, eta):
    #training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    data_set, width, height, resolution = camera_loader.simple_data_set(n)
    data_set, input_size, output_size = loader(os.path.join(dirname, "TestData"))
    net = nnmodel.Network([resolution, resolution, resolution])
    for i in range(n):
        data_set, width, height, resolution = camera_loader.simple_data_set(n)
        net.SGD(data_set, 10, 10, eta, test_data=None)
    
    cap = cv2.VideoCapture(0)
    
    while True:
        # capture frame
        ret, frame = cap.read()
        dim = (width, height)
        frame = cv2.resize(frame, dim, interpolation=cv2.INTER_NEAREST)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 0)
        test = np.array(gray).reshape([resolution, 1])
        result = net.feedforward(test)
        result = np.reshape(result, (-1, width))
        im = np.array(result * 255, dtype=np.uint8)
        cv2.imshow('real', gray)
        cv2.imshow('generated', im)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    
