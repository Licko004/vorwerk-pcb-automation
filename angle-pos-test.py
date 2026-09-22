import os
os.environ["QT_QPA_PLATFORM"] = "wayland"

import cv2
import numpy as np
import time

from picamera2 import Picamera2
picam2 = Picamera2()

picam2.preview_configuration.main.size = (1280,720)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")

picam2.start()
time.sleep(2)
img  = picam2.capture_file("test.png")
time.sleep(1)

img = cv2.imread("test.png")
hh, ww, cc = img.shape

# convert to gray
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imwrite("grayscale.png", gray)

# threshold the grayscale image
ret1, thresh1 = cv2.threshold(gray,127,255,cv2.THRESH_TRUNC)
ret2, thresh2 = cv2.threshold(gray,127,255,cv2.THRESH_TOZERO)
ret3, thresh3 = cv2.threshold(gray,127,255,cv2.THRESH_TOZERO_INV)
ret4, thresh4 = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)


cv2.imwrite("threshold-trunc.jpg", thresh1)
cv2.imwrite("threshold-tozero.png", thresh2)
cv2.imwrite("threshold-tozero-inv.png", thresh3)
cv2.imwrite("threshold-binary.png", thresh4)

im2, contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(thresh1, contours, -1, (0,255,0), 3)

cv2.imwrite("contours.png", thresh1)

# # find outer contour
# cntrs = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cntrs = cntrs[0] if len(cntrs) == 2 else cntrs[1]

# # get rotated rectangle from outer contour
# rotrect = cv2.minAreaRect(cntrs[0])
# box = cv2.boxPoints(rotrect)
# box = np.int8(box)

# # draw rotated rectangle on copy of img as result
# result = img.copy()
# cv2.drawContours(result,[box],0,(0,0,255),2)

# # get angle from rotated rectangle
# angle = rotrect[-1]

# # from https://www.pyimagesearch.com/2017/02/20/text-skew-correction-opencv-python/
# # the `cv2.minAreaRect` function returns values in the
# # range [-90, 0); as the rectangle rotates clockwise the
# # returned angle trends to 0 -- in this special case we
# # need to add 90 degrees to the angle
# # if angle < -45:
# #     angle = -(90 + angle)
 
# # # otherwise, just take the inverse of the angle to make
# # # it positive
# # else:
# #     angle = -angle

# print(angle,"deg")

# # write result to disk
# cv2.imwrite("wing2_rotrect.png", result)

# cv2.imshow("THRESH", thresh)
# cv2.imshow("RESULT", result)
cv2.waitKey(0)
cv2.destroyAllWindows()



