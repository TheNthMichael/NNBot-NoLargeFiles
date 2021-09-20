is_recording = False
is_not_exiting = True

screen_cap_scale = 8

FPS = 20

HISTORY_LENGTH = 10

MAX_CHUNK_SIZE = 10000000

KEY_THRESHOLD = 0.5

monitor_region = {'top': 0, 'left': 0, 'width': 1920, 'height': 1080}

screen_cap_sizes = ((monitor_region['width'] - monitor_region['left']) // screen_cap_scale, (monitor_region['height'] - monitor_region['top']) // screen_cap_scale)

screen_cap_resolution = screen_cap_sizes[0] * screen_cap_sizes[1]