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

picam2.preview_configuration.main.size = (1920,1080)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")

PIXEL_PIN = board.D18       # pin that the NeoPixel ring is connected to
NUM_PIXELS = 24              
ORDER = neopixel.GRBW       
BRIGHTNESS = 1.0             # full brightness

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

# LEDRING ON sequence:
# pixels = neopixel.NeoPixel(PIXEL_PIN, NUM_PIXELS, brightness=BRIGHTNESS, pixel_order=ORDER)

# pixels.fill((0, 0, 0, 255))   # R, G, B, W
# pixels.show()

picam2.start()
img  = picam2.capture_file("raw-png-PCA.png")

img = cv.imread("raw-png-PCA.png")

# apply HSV and GRAYSCALE to image
# hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Thresholding white region
ret_white, thresh_white = cv.threshold(gray,0,255,cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
# cv.imwrite("white_mask.png", thresh_white)

contours, _ = cv.findContours(thresh_white, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
contour_img = img.copy()

cntr = []
eigenvectors = []
for i, c in enumerate(contours):
    # Calculate the area of each contour
    area = cv.contourArea(c)
    #print("Contours area: ", area)
    
    # Ignore contours that are too small or too large
    if area < 2000 or 4000 < area:
        continue

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
    
    DPI_X = 5.749716133480659 
    x_offset_wrld = 17.5 # 15.7mm measured using calipers
    x_offset_pix = DPI_X*x_offset_wrld
    
    pick_point = cntr - x_offset_pix*x_axis 
    
    cv.circle(img, tuple(pick_point.astype(int)), 5, (0, 255, 0), -1)
    # print("Center:", cntr)
    print("Pick point:", pick_point)
    print("Angle: ", tilt_deg)

cv.imwrite("PCA-result.png", img)
cv.imwrite("PCA-contours.png", contour_img)

# LEDRING OFF sequence:
# pixels.fill((0, 0, 0, 0))   
# pixels.show()