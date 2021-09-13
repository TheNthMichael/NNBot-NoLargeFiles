import cv2
import time
import subprocess
import numpy as np

w,h = 1920, 1080
fps = 60

def ffmpegGrab():
    """Generator to read frames from ffmpeg subprocess"""
    # Use "-f rawvideo" instead of "-f image2pipe" (command is working with image2pipe, but rawvideo is the correct format).
    cmd = f'.\\Resources\\ffmpeg.exe -f gdigrab -framerate {fps} -offset_x 0 -offset_y 0 -video_size {w}x{h} -i desktop -pix_fmt bgr24 -vcodec rawvideo -an -sn -f rawvideo -'

    #proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)

    # Don't use stderr=subprocess.STDOUT, and don't use shell=True
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    while True:
        raw_frame = proc.stdout.read(w*h*3)
        frame = np.frombuffer(raw_frame, np.uint8)  # Use frombuffer instead of fromarray
        frame = frame.reshape((h, w, 3))
        yield frame

# Get frame generator
gen = ffmpegGrab()

# Get start time
start = time.time()

# Read video frames from ffmpeg in loop
nFrames = 0
while True:
    # Read next frame from ffmpeg
    frame = next(gen)
    nFrames += 1

    frame = cv2.resize(frame, (w // 4, h // 4))

    cv2.imshow('screenshot', frame)

    if cv2.waitKey(1) == ord("q"):
        break

    fps = nFrames/(time.time()-start)
    print(f'FPS: {fps}')


cv2.destroyAllWindows()