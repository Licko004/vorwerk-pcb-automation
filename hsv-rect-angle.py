import os
os.environ["QT_QPA_PLATFORM"] = "wayland"

import cv2 as cv2
import imutils
import numpy as np
import time
import board
import neopixel
import math

from picamera2 import Picamera2
picam2 = Picamera2()

picam2.preview_configuration.main.size = (1920,1080)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")

PIXEL_PIN = board.D18       # pin that the NeoPixel ring is connected to
NUM_PIXELS = 24              # your ring has 24 LEDs
ORDER = neopixel.GRBW        # RGBW ring
BRIGHTNESS = 1.0             # full brightness

pixels = neopixel.NeoPixel(PIXEL_PIN, NUM_PIXELS, brightness=BRIGHTNESS, pixel_order=ORDER)

pixels.fill((0, 0, 0, 255))   # R, G, B, W
pixels.show()

picam2.start()
img  = picam2.capture_file("test.png")

img = cv2.imread("test.png")

# apply HSV to the image rather then grayscale
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# isolate green (PCB body) by hue -- tune these ranges to your actual PCB color
lower_green = np.array([25, 40, 20])
upper_green = np.array([100, 255, 255])
green_mask = cv2.inRange(hsv, lower_green, upper_green)
cv2.imwrite("green_mask.png", green_mask)

# isolate bright/white regions (connector) by low saturation + high value
# lower_white = np.array([10, 0, 180])
# upper_white = np.array([210, 60, 255])
# white_mask = cv2.inRange(hsv, lower_white, upper_white)
# cv2.imwrite("white_mask.png", white_mask)
ret_white, thresh_white = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
cv2.imwrite("white_mask.png", thresh_white)

# combine: whole PCB = green body OR white connector
# pcb_mask = cv2.bitwise_or(green_mask, white_mask)

# pcb_cnts, _ = cv2.findContours(pcb_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_L1)
# pcb_cnts = imutils.grab_contours((pcb_cnts, _))

# contour_unfiltered = img.copy()
# cv2.drawContours(contour_unfiltered, pcb_cnts, -1, (0, 255, 0), 2)
# cv2.imwrite("HSV-contours-unfiltered.png", contour_unfiltered)

pixels.fill((0, 0, 0, 0))   # all channels off (use (0, 0, 0) if your ring is RGB, not RGBW)
pixels.show()