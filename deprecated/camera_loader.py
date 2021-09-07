import numpy as np
import gzip
import cv2

"""
def collect_data(size_training, num_predicting):
    cap = cv2.VideoCapture(0)
    frames = []
    training_inputs = []
    training_results = []
    for _ in range(size_training+num_predicting):
        # capture frame
        ret, frame = cap.read()
        # operate on frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)
    
    for i in range(size_training):
        training_inputs.append(frames[i])
        training_result = []
        for j in range(num_predicting):
            print('a')
"""

def simple_data_set(size_training):
    cap = cv2.VideoCapture(0)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) // 5
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) // 5
    resolution = width * height

    print("res=", resolution)

    training_inputs = []
    training_results = []

    for _ in range(size_training):
        # capture frame
        ret, frame = cap.read()
        dim = (width, height)
        new = cv2.resize(frame, dim, interpolation=cv2.INTER_NEAREST)
        # operate on frame
        gray = cv2.cvtColor(new, cv2.COLOR_BGR2GRAY)
        gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 0)
        print(gray)
        # reshape
        training_inputs.append(np.array(gray).reshape([resolution, 1]))
        training_results.append(np.array(gray).reshape([resolution, 1]))
    cap.release()
    data_set = zip(training_inputs, training_results)
    return (data_set, width, height, resolution)



"""
cap = cv2.VideoCapture(0)

while True:
    # capture frame
    ret, frame = cap.read()

    # operate on frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    for i in range(100):
        for j in range(100):
            gray[i][j] = 0

    # display
    cv2.imshow('frame', gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# cleanup
cap.release()
cv2.destroyAllWindows()
"""
