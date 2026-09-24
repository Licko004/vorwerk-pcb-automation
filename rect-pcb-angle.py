import os
os.environ["QT_QPA_PLATFORM"] = "wayland"

import cv2 as cv2
import imutils
import numpy as np
import time
import board
import neopixel
import math

# Configure the setup
# PIXEL_PIN = board.D18       # pin that the NeoPixel ring is connected to
# NUM_PIXELS = 24              # your ring has 24 LEDs
# ORDER = neopixel.GRBW        # RGBW ring
# BRIGHTNESS = 1.0             # full brightness

# pixels = neopixel.NeoPixel(PIXEL_PIN, NUM_PIXELS, brightness=BRIGHTNESS, pixel_order=ORDER)

# pixels.fill((0, 0, 0, 255))   # R, G, B, W
# pixels.show()

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

# Source - https://stackoverflow.com/a/59238613
# Posted by chamith mawela
# Retrieved 2026-09-23, License - CC BY-SA 4.0
cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_L1)
cnts = imutils.grab_contours(cnts)

contour_unfiltered = img.copy()
cv2.drawContours(contour_unfiltered, cnts, -1, (0, 255, 0), 2)
cv2.imwrite("contours-unfiltered.png", contour_unfiltered) 

cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
rect_areas = []
for c in cnts: # exclude largest found contour
    (x, y, w, h) = cv2.boundingRect(c)
    rect_areas.append(w * h)
    
avg_area = np.average(rect_areas)
print("avg area:", avg_area)
rect_drawn = img.copy()

for c in cnts:
    (x, y, w, h) = cv2.boundingRect(c)
    cv2.rectangle(rect_drawn, (x, y), (x + w, y + h), (0, 0, 255), 2)
cv2.imwrite("contours-rectangles.png", rect_drawn)
    
big_cnts = []
for c in cnts:
    (x, y, w, h) = cv2.boundingRect(c)
    cnt_area = w * h
    print("Unfiltered areas:", cnt_area)
    if cnt_area > 0.30 * avg_area and cnt_area < 0.80 * avg_area:
        # thresh_filtered[y:y + h, x:x + w] = 0
        big_cnts.append(c)
        
        
contour_filtered = img.copy()
cv2.drawContours(contour_filtered, big_cnts, -1, (0, 255, 0), 2) #filter contours larger then 20% of average area
cv2.imwrite("contours-filtered.png", contour_filtered) #create an image with the filtered contours drawn

# Draw rectangle around pcb contours only
for c in big_cnts:
    (x, y, w, h) = cv2.boundingRect(c)
    fil_rect_area = w * h
    # Draw dot in center of rectangles
    cx = x + w/2
    cy = y + h/2
    cx_int = int(cx)
    cy_int = int(cy)
    print("Rectangle center positon: ", cx,",",cy)
    points_img = cv2.circle(contour_filtered, (cx_int,cy_int), 5, color = (0,0,255), thickness = -1)
    print("Rect_area",fil_rect_area)
    if fil_rect_area > 4000 and fil_rect_area < 8000: #filtering by pixel area of contours
        # cv2.rectangle(contour_filtered, (x, y), (x + w, y + h), (0, 0, 255), 2) # Draws rectangle around pcb, also including spikes etc
        rotrect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rotrect)
        box = np.int32(box)
        # Draw line at rotrect angle of length 18mm
        line_length = 17
        line_angle_def = rotrect[-1]

        # find the longer edge of the rectangle, to compute the angle
        edge1 = box[1] - box[0]
        edge2 = box[2] - box[1]
        
        if np.linalg.norm(edge1) > np.linalg.norm(edge2):
            long_edge = edge1
        else:
            long_edge = edge2
        
        
        long_edge_angle = math.degrees(math.atan2(long_edge[1], long_edge[0]))
        line_angle = long_edge_angle + 90   # perpendicular to the long edge
        
        DPI = 6.279251163812498 # gathered from DPI_calibration.py
        pixel_legth = line_length*DPI
        print("rectangle angle: ", line_angle_def)
        
        line_angle = long_edge_angle + 90
        print("line angle: ", line_angle)
        
        angle_radians = math.radians(line_angle)
        
        x2 = int(cx + pixel_legth * math.cos(angle_radians))
        y2 = int(cy + pixel_legth * math.sin(angle_radians))
        
        cv2.line(contour_filtered, (cx_int, cy_int), (x2, y2), color = (0,255,0), thickness = 3)
        
        cv2.drawContours(contour_filtered, [box], 0, (0, 0, 255), 2)      

    
cv2.imwrite("rect-fil-contours.png", contour_filtered)

# pixels.fill((0, 0, 0, 0))   # all channels off (use (0, 0, 0) if your ring is RGB, not RGBW)
# pixels.show()

