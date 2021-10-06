from threading import Event, Lock
import cv2

is_recording = Event()
is_exiting = Event()

def toggle_recording():
    if is_recording.is_set():
        is_recording.clear()
    else:
        is_recording.set()

#is_recording = False
#is_not_exiting = True

screen_cap_scale = 10

def crop_to_new_aspect_ratio(img, img_dim: tuple, aspect_ratio: tuple, end_size: tuple):
    """Returns the cropped and resized image."""
    w = img_dim[0]
    h = img_dim[1]
    aw = aspect_ratio[0]
    ah = aspect_ratio[1]
    x = w // aw
    if not (aw * x <= w and ah * x <= h):
        x = h // ah
        assert(aw * x <= w and ah * x <= h)
    h_space = (h - (ah * x)) // 2
    w_space = (w - (aw * x)) // 2
    roi = img[h_space:(h-h_space), w_space:(w - w_space)]
    return cv2.resize(roi, end_size)

FPS = 10

HISTORY_LENGTH = FPS

MAX_CHUNK_SIZE = 10000000
#MAX_CHUNK_SIZE = 2000

KEY_THRESHOLD = 0.6

monitor_region = {'top': 0, 'left': 0, 'width': 1920, 'height': 1080}

screen_cap_sizes = ((monitor_region['width'] - monitor_region['left']) // screen_cap_scale, (monitor_region['height'] - monitor_region['top']) // screen_cap_scale)

screen_cap_sizes_unscaled = ((monitor_region['width'] - monitor_region['left']), (monitor_region['height'] - monitor_region['top']))

screen_cap_resolution = screen_cap_sizes[0] * screen_cap_sizes[1]