#!/usr/bin/env python3

# ffmpeg -y -pix_fmt bgr0 -f avfoundation -r 20 -t 10 -i 1 -vf scale=w=3840:h=2160 -f rawvideo /dev/null

import sys
import os
import msvcrt
import cv2
import time
import subprocess
import numpy as np


w,h = 1920, 1080
fps = 60

#out = cv2.VideoWriter('output.avi', -1, 20.0, (w,h))

def ffmpegGrab():
    """Generator to read frames from ffmpeg subprocess"""
    cmd = f'.\\Resources\\ffmpeg.exe -f gdigrab -framerate {fps} -video_size {w}x{h} -i desktop -pix_fmt bgr24 -c:v h264_nvenc -qp 0 -vcodec rawvideo -an -sn -f image2pipe pipe:1' 

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=False)
    msvcrt.setmode(proc.stdout.fileno(), os.O_BINARY)
    #out, err = proc.communicate()
    while True:
        raw_frame = proc.stdout.read(w*h*3)
        frame = np.frombuffer(raw_frame, np.uint8)
        frame = frame.reshape((h, w, 3))
        yield frame

# Get frame generator
gen = ffmpegGrab()

# Get start time
start = time.time()
cmd = f'.\\Resources\\ffmpeg.exe -f gdigrab -framerate {fps} -video_size {w}x{h} -i desktop -pix_fmt bgr24 -c:v h264_nvenc -qp 0 -vcodec rawvideo -an -sn -f image2pipe pipe:1' 

proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, shell=False)
#msvcrt.setmode(proc.stdout.fileno(), os.O_BINARY)
# Read video frames from ffmpeg in loop
nFrames = 0
while True:
    # Read next frame from ffmpeg
    raw_frame = proc.stdout.read(w*h*3)
    frame = np.frombuffer(raw_frame, np.uint8)
    frame = frame.reshape((h, w, 3))
    #frame = next(gen)
    #cv2.imwrite(f"test/sample{nFrames}.png", frame)
    nFrames += 1

    frame = cv2.resize(frame, (w // 4, h // 4))

    #out.write(frame)

    cv2.imshow('screenshot', frame)

    if cv2.waitKey(1) == ord("q"):
        break

    fps = nFrames/(time.time()-start)
    print(f'FPS: {fps}')


cv2.destroyAllWindows()