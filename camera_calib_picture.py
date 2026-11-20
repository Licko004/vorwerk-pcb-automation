import os
os.environ["QT_QPA_PLATFORM"] = "xcb"

import re
import time

import cv2 as cv
import board
import neopixel
from picamera2 import Picamera2

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
FOLDER = "cam_calib_pictures"              # folder the images are saved into
PREFIX = "img"                 # img1.png, img2.png, ...
EXT = ".png"

RESOLUTION = (3280, 2464)      # saved image size (full IMX219 resolution)
PREVIEW_SIZE = (640, 480)      # live preview only, must have the same 4:3 aspect ratio

USE_LED = True                 # light the ring while capturing
PIXEL_PIN = board.D18
NUM_PIXELS = 24
ORDER = neopixel.GRBW
BRIGHTNESS = 1.0

WINDOW = "Capture  (SPACE = save, Q / Esc = quit)"


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def next_index(folder):
    """Next free number: one above the highest imgN.png already in the folder,
    so existing images are never overwritten, even across runs."""
    pattern = re.compile(rf"^{re.escape(PREFIX)}(\d+){re.escape(EXT)}$")
    numbers = []
    for name in os.listdir(folder):
        m = pattern.match(name)
        if m:
            numbers.append(int(m.group(1)))
    return max(numbers, default=0) + 1


# ----------------------------------------------------------------------------
# Setup
# ----------------------------------------------------------------------------
os.makedirs(FOLDER, exist_ok=True)         # does nothing if the folder already exists
counter = next_index(FOLDER)
print(f"Saving to '{FOLDER}', next image: {PREFIX}{counter}{EXT}")

picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"size": RESOLUTION, "format": "RGB888"},       # full-resolution frame that gets saved
    lores={"size": PREVIEW_SIZE, "format": "YUV420"},    # small frame for the live view
)
picam2.configure(config)

pixels = None
if USE_LED:
    pixels = neopixel.NeoPixel(PIXEL_PIN, NUM_PIXELS,
                               brightness=BRIGHTNESS, pixel_order=ORDER)

# ----------------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------------
try:
    if USE_LED:
        pixels.fill((0, 0, 0, 255))        # R, G, B, W
        pixels.show()

    picam2.start()
    time.sleep(2)                          # let exposure / white balance settle

    saved = 0
    while True:
        yuv = picam2.capture_array("lores")
        preview = cv.cvtColor(yuv, cv.COLOR_YUV420p2BGR)
        cv.putText(preview, f"saved: {saved}   next: {PREFIX}{counter}{EXT}", (10, 25),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv.LINE_AA)
        cv.imshow(WINDOW, preview)

        key = cv.waitKey(1) & 0xFF         # the preview window must have keyboard focus
        if key == 32:                      # space
            frame = picam2.capture_array("main")
            path = os.path.join(FOLDER, f"{PREFIX}{counter}{EXT}")
            if cv.imwrite(path, frame):
                print(f"Saved {path}  {frame.shape[1]}x{frame.shape[0]}")
                counter += 1
                saved += 1
            else:
                print(f"Could not write {path}")
        elif key in (ord("q"), 27):        # q or Esc
            break

finally:
    cv.destroyAllWindows()
    picam2.stop()
    if USE_LED:
        pixels.fill((0, 0, 0, 0))
        pixels.show()

print(f"Done, {saved} image(s) saved.")