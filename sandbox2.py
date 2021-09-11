import sys
import cv2
import time
import subprocess
import numpy as np

import d3dshot

w,h = 1920, 1080


# Get start time
start = time.time()

d = d3dshot.create(capture_output="numpy", frame_buffer_size=w * h * 3)
d.capture()

cv2.namedWindow('player')

# Read video frames from ffmpeg in loop
nFrames = 0
while True:
    # Read next frame from ffmpeg
    frame = d.get_latest_frame()
    if frame is None:
        continue
    nFrames += 1
    frame = cv2.resize(frame,(960,540))
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    cv2.imshow('player', frame)

    if cv2.waitKey(1) == ord("q"):
        break

    fps = nFrames/(time.time()-start)
    print(f'FPS: {fps}')


cv2.destroyAllWindows()
d.stop()