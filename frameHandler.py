import subprocess
from typing_extensions import TypeVarTuple
import numpy as np
from threading import Event, Lock

# This will be run in another thread outside the main loop so that the main loop can control when frames are pulled
class FrameHandler:
    def __init__(self, region, fps) -> None:
        self.fps = fps
        self.region = region
        self.frame_width = self.region["width"] - self.region["left"]
        self.frame_height = self.region["height"] - self.region["top"]
        self.resolution = self.frame_height * self.frame_width
        self.num_channels = 3
        self.cmd = f'.\\Resources\\ffmpeg.exe -f gdigrab -framerate {self.fps} -offset_x {self.region["left"]} -offset_y {self.region["top"]} -video_size {self.region["width"]}x{self.region["height"]} -i desktop -pix_fmt bgr24 -vcodec rawvideo -an -sn -f rawvideo -'
        # Don't use stderr=subprocess.STDOUT, and don't use shell=True
        self.proc = subprocess.Popen(self.cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self.last_frame = None

    def update(self):
        raw_frame = self.proc.stdout.read(self.resolution * self.num_channels)
        frame = np.frombuffer(raw_frame, np.uint8)  # Use frombuffer instead of fromarray
        self.last_frame = frame.reshape((self.frame_height, self.frame_width, self.num_channels))

    def get_current_frame(self):
        return self.last_frame


def frame_handler_thread(frame_handler: FrameHandler, exit_event, data_lock=None):
    try:
        while not exit_event.is_set():
            frame_handler.update()
    except Exception as e:
        print(e)
        exit_event.set()
    finally:
        print("Closing frame_handler_thread...")


    