import os
os.environ["QT_QPA_PLATFORM"] = "wayland"

import cv2 as cv2
import imutils
import numpy as np
import time
import board
import neopixel
import math

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
PIXEL_PIN = board.D18        # pin that the NeoPixel ring is connected to
NUM_PIXELS = 24              # your ring has 24 LEDs
ORDER = neopixel.GRBW        # RGBW ring
BRIGHTNESS = 0.5             # 1 = full brightness

# Size filter on the bounding box (w * h, in pixels at 1920x1080).
# From your printout the three parts were 5782, 4900, 4900, the next largest
# non-part was 1332 and the tray was 913123, so this window isolates the parts.
MIN_AREA = 3500
MAX_AREA = 8000

# GREEN PCB FILTER SETTINGS
GREEN_LO = (55, 60, 40)
GREEN_HI = (90, 255, 255)
MIN_BOARD_AREA = 25000

# Tray detection limits
TRAY_ERODE = 15   

DPI = 6.279251163812498      # pixels per mm, from DPI_calibration.py
LINE_LENGTH_MM = 17

# ----------------------------------------------------------------------------
# Hardware
# ----------------------------------------------------------------------------
pixels = neopixel.NeoPixel(PIXEL_PIN, NUM_PIXELS, brightness=BRIGHTNESS, pixel_order=ORDER)

from picamera2 import Picamera2
picam2 = Picamera2()

picam2.preview_configuration.main.size = (1920, 1080)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")

try:
    pixels.fill((0, 0, 0, 255))   # R, G, B, W
    pixels.show()

    picam2.start()
    picam2.capture_file("test.png")

finally:
    picam2.stop()
    pixels.fill((0, 0, 0, 0))     # all channels off
    pixels.show()

img = cv2.imread("test.png")


# ----------------------------------------------------------------------------
# Threshold and contours
# ----------------------------------------------------------------------------
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Inverted Otsu: the tray is white, the connectors are holes inside it
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
cv2.imwrite("threshold-otsu-connector.png", thresh)

cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_L1)
cnts = imutils.grab_contours(cnts)

# Seperate the tray from other parts
tray = max(cnts, key=cv2.contourArea)

tray_mask = np.zeros_like(gray)
cv2.drawContours(tray_mask, [tray], -1, 255, cv2.FILLED)
tray_mask = cv2.erode(tray_mask, np.ones((TRAY_ERODE, TRAY_ERODE), np.uint8))

contour_unfiltered = img.copy()
cv2.drawContours(contour_unfiltered, cnts, -1, (0, 255, 0), 2)
cv2.imwrite("contours-unfiltered.png", contour_unfiltered)

green = cv2.bitwise_and(cv2.inRange(hsv, GREEN_LO, GREEN_HI), tray_mask)
cv2.imwrite("green_mask.png", green)
green_contours, _ = cv2.findContours(green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

greenPCB_contours = img.copy()
cv2.drawContours(greenPCB_contours, cnts, -1, (255, 0, 0), 3)
cv2.imwrite("contours-PCB.png", greenPCB_contours)

# ----------------------------------------------------------------------------
# Size filter (absolute window, no average)
# ----------------------------------------------------------------------------
part_cnts = []
for c in cnts:
    (x, y, w, h) = cv2.boundingRect(c)
    area = w * h
    if area >= 500:                       # print only meaningful blobs, not pixel dust
        print(f"bbox area={area}  at ({x},{y}) size {w}x{h}")
    if MIN_AREA < area < MAX_AREA:
        part_cnts.append(c)

print(f"\nContours passing the size filter: {len(part_cnts)} (expected 3)")

contour_filtered = img.copy()
cv2.drawContours(contour_filtered, part_cnts, -1, (0, 255, 0), 2)
cv2.imwrite("contours-filtered.png", contour_filtered)

# ----------------------------------------------------------------------------
# Centre, rotated rectangle and angle line for each part
# ----------------------------------------------------------------------------
for c in part_cnts:
    (x, y, w, h) = cv2.boundingRect(c)

    # Dot in the centre of the bounding rectangle
    cx = x + w / 2
    cy = y + h / 2
    cx_int, cy_int = int(cx), int(cy)
    print("Rectangle center position: ", cx, ",", cy)
    cv2.circle(contour_filtered, (cx_int, cy_int), 5, (0, 0, 255), -1)

    rotrect = cv2.minAreaRect(c)
    box = np.int32(cv2.boxPoints(rotrect))
    print("rectangle angle: ", rotrect[-1])

    # Find the longer edge of the rectangle to compute the angle
    edge1 = box[1] - box[0]
    edge2 = box[2] - box[1]
    long_edge = edge1 if np.linalg.norm(edge1) > np.linalg.norm(edge2) else edge2

    long_edge_angle = math.degrees(math.atan2(long_edge[1], long_edge[0]))
    line_angle = long_edge_angle + 90          # perpendicular to the long edge
    print("line angle: ", line_angle)

    angle_radians = math.radians(line_angle)
    pixel_length = LINE_LENGTH_MM * DPI
    x2 = int(cx + pixel_length * math.cos(angle_radians))
    y2 = int(cy + pixel_length * math.sin(angle_radians))

    cv2.line(contour_filtered, (cx_int, cy_int), (x2, y2), color=(0, 255, 0), thickness=3)
    cv2.drawContours(contour_filtered, [box], 0, (0, 0, 255), 2)

cv2.imwrite("rect-fil-contours.png", contour_filtered)