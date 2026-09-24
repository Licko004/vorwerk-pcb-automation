import os
os.environ["QT_QPA_PLATFORM"] = "wayland"

import cv2 as cv2
import imutils
import numpy as np
import time
import board
import neopixel

# Configure the setup
PIXEL_PIN = board.D18       # pin that the NeoPixel ring is connected to
NUM_PIXELS = 24              # your ring has 24 LEDs
ORDER = neopixel.GRBW        # RGBW ring
BRIGHTNESS = 1.0             # full brightness

pixels = neopixel.NeoPixel(PIXEL_PIN, NUM_PIXELS, brightness=BRIGHTNESS, pixel_order=ORDER)

pixels.fill((0, 0, 0, 255))   # R, G, B, W
pixels.show()

from picamera2 import Picamera2
picam2 = Picamera2()

picam2.preview_configuration.main.size = (1920,1080)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")

picam2.start()
img  = picam2.capture_file("test.png")
img = cv2.imread("test.png")


# convert to gray
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imwrite("grayscale-noise.png", gray)

#Thresholding the grayscale image
ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
# cv2.imwrite("threshold-otsu.png", thresh)

cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_L1)
cnts = imutils.grab_contours(cnts)

contour_unfiltered = img.copy()
cv2.drawContours(contour_unfiltered, cnts, -1, (0, 255, 0), 2)
cv2.imwrite("DPI-contours.png", contour_unfiltered) 

cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
rect_areas = []
for c in cnts:
    (x, y, w, h) = cv2.boundingRect(c)
    rect_areas.append(w * h)
rotrect = cv2.minAreaRect(cnts[0])
(tray_x, tray_y), (tray_w, tray_h), angle = rotrect
print("rotrect:", rotrect)

tray_real_width_mm = 146    # replace with your tray's actual width
tray_real_height_mm = 186   # replace with your tray's actual height

# pixels per mm
px_per_mm_x = tray_w / tray_real_width_mm
px_per_mm_y = tray_h / tray_real_height_mm
print("pix_per_mm_X", px_per_mm_x)
print("pix_per_mm_Y", px_per_mm_y)

pixels.fill((0, 0, 0, 0))   # all channels off (use (0, 0, 0) if your ring is RGB, not RGBW)
pixels.show()
