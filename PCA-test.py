import os
os.environ["QT_QPA_PLATFORM"] = "wayland"

import cv2 as cv
import imutils
import numpy as np
import time
import board
import neopixel
import math
from math import atan2, cos, sin, sqrt, pi


from picamera2 import Picamera2
picam2 = Picamera2()

config = picam2.create_still_configuration(
    main={"size": (3280, 2464), "format": "RGB888"},
    raw={"size": (3280, 2464)},
)

picam2.preview_configuration.main.size = (3280,2464)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure(config)

PIXEL_PIN = board.D18       # pin that the NeoPixel ring is connected to
NUM_PIXELS = 24              
ORDER = neopixel.GRBW       
BRIGHTNESS = 1.0             # full brightness

# Connector dimensions, filtering parameters
LONG_NOM, SHORT_NOM = 105, 52
TOL = 0.2
MIN_FILL = 0.8

# Helper functions:

# Get orientation of certain contour using PCA
def getOrientation(pts, img):
    
    sz = len(pts)
    data_pts = np.empty((sz, 2), dtype=np.float64)
    for i in range(data_pts.shape[0]):
        data_pts[i,0] = pts[i,0,0]
        data_pts[i,1] = pts[i,0,1]
 
    # Perform PCA analysis
    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv.PCACompute2(data_pts, mean)
 
    # Store the center of the object
    cntr = (int(mean[0,0]), int(mean[0,1]))

    cv.circle(img, cntr, 3, (255, 0, 255), 2)
    p1 = (cntr[0] + 0.02 * eigenvectors[0,0] * eigenvalues[0,0], cntr[1] + 0.02 * eigenvectors[0,1] * eigenvalues[0,0])
    p2 = (cntr[0] - 0.02 * eigenvectors[1,0] * eigenvalues[1,0], cntr[1] - 0.02 * eigenvectors[1,1] * eigenvalues[1,0])
    drawAxis(img, cntr, p1, (0, 255, 0), 1)
    drawAxis(img, cntr, p2, (255, 255, 0), 5)
 
    angle = atan2(eigenvectors[0,1], eigenvectors[0,0]) # orientation in radians
    
 
    return angle, cntr, eigenvectors

# Draw axis on calculated PCA center
def drawAxis(img, p_, q_, colour, scale):
    p = list(p_)
    q = list(q_)
    
    angle = atan2(p[1] - q[1], p[0] - q[0]) # angle in radians
    hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))
 
    # Here we lengthen the arrow by a factor of scale
    q[0] = p[0] - scale * hypotenuse * cos(angle)
    q[1] = p[1] - scale * hypotenuse * sin(angle)
    cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv.LINE_AA)
 
    # create the arrow hooks
    p[0] = q[0] + 9 * cos(angle + pi / 4)
    p[1] = q[1] + 9 * sin(angle + pi / 4)
    cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv.LINE_AA)
 
    p[0] = q[0] + 9 * cos(angle - pi / 4)
    p[1] = q[1] + 9 * sin(angle - pi / 4)
    cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv.LINE_AA)

# Connector filter    
def is_connector(c):
    (cx, cy), (w, h), _ = cv.minAreaRect(c)
    long_s, short_s = max(w, h), min(w, h)
    if short_s == 0:
        return False
    fill = cv.contourArea(c) / (long_s * short_s)
    return (abs(long_s - LONG_NOM) < TOL * LONG_NOM and
            abs(short_s - SHORT_NOM) < TOL * SHORT_NOM and
            fill > MIN_FILL)

# LEDRING ON sequence:
pixels = neopixel.NeoPixel(PIXEL_PIN, NUM_PIXELS, brightness=BRIGHTNESS, pixel_order=ORDER)

pixels.fill((0, 0, 0, 255))   # R, G, B, W
pixels.show()

picam2.start()

# time.sleep(2)

img  = picam2.capture_file("raw-png-PCA.png")
img = cv.imread("raw-png-PCA.png")

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
blur = cv.GaussianBlur(gray, (5, 5), 0)

# 1) Tray mask: the tray is the largest dark blob (Otsu, inverted)
_, dark = cv.threshold(blur, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
cnts, _ = cv.findContours(dark, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
tray = max(cnts, key=cv.contourArea)
tray_mask = np.zeros_like(gray)
cv.drawContours(tray_mask, [tray], -1, 255, cv.FILLED)   # filled, so connectors inside are kept
tray_mask = cv.erode(tray_mask, np.ones((15, 15), np.uint8))  # stay off the tray edge

# 2) Bright connector body inside the tray only
WHITE_T = 120   # tune: connector is much brighter than the tray and the green PCB
_, bright = cv.threshold(blur, WHITE_T, 255, cv.THRESH_BINARY)
thresh_white = cv.bitwise_and(bright, tray_mask)

# 3) Clean up: open removes specks and thin pin glints, close fills holes
thresh_white = cv.morphologyEx(thresh_white, cv.MORPH_OPEN,
                               cv.getStructuringElement(cv.MORPH_RECT, (35, 35)))
thresh_white = cv.morphologyEx(thresh_white, cv.MORPH_CLOSE,
                               cv.getStructuringElement(cv.MORPH_RECT, (15, 15)))
cv.imwrite("white_mask.png", thresh_white)

# 4) Now RETR_EXTERNAL is safe, since the paper is masked out
contours, _ = cv.findContours(thresh_white, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

for c in contours:
    a = cv.contourArea(c)
    (cx, cy), (w, h), _ = cv.minAreaRect(c)
    print(f"area={a:.0f} center=({cx:.0f},{cy:.0f}) long={max(w,h):.0f} short={min(w,h):.0f}")

# Find PCB connector contours
cntr = []
eigenvectors = []
for i, c in enumerate(contours):
    if not is_connector(c):
        continue
    # Calculate the area of each contour
    area = cv.contourArea(c)

    # Draw each contour only for visualisation purposes
    cv.drawContours(img, contours, i, (0, 0, 255), 2)
    # Find the orientation of each shape
    angle, cntr, eigenvectors = getOrientation(c, img)
    angle_deg = math.degrees(angle)
    
    # Translate cntr point on the X axis:
    cntr = np.array(cntr, dtype=float)
    eigenvectors = np.array(eigenvectors, dtype=float)
    
    v = eigenvectors[0].copy()      # long axis
    if v[1] < 0:                    # force it to point down the image (+y)
        v = -v

    tilt_deg = math.degrees(atan2(v[0], v[1]))   # 0 = perfectly vertical
    
    # X-axis orientation to give angle the right sign
    x_axis = eigenvectors[1].copy()
    if x_axis[0] < 0:               # always point toward +x in the image
        x_axis = -x_axis
    
    DPI_X = 17.249447635881573
    x_offset_wrld = 17.5 # 15.7mm measured using calipers
    x_offset_pix = DPI_X*x_offset_wrld
    
    pick_point = cntr - x_offset_pix*x_axis 
    
    cv.circle(img, tuple(pick_point.astype(int)), 5, (0, 255, 0), -1)
    # print("Center:", cntr)
    print("Pick point:", pick_point)
    print("Angle: ", tilt_deg)

cv.imwrite("PCA-result.png", img)

# LEDRING OFF sequence:
pixels.fill((0, 0, 0, 0))   
pixels.show()