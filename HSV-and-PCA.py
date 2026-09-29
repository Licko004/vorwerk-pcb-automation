import os
os.environ["QT_QPA_PLATFORM"] = "wayland"

import time
import math
from math import atan2, cos, sin, sqrt, pi

import cv2 as cv
import numpy as np
import board
import neopixel
from picamera2 import Picamera2

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
USE_LED = False                 # keep identical to the lighting used while tuning

PIXEL_PIN = board.D18
NUM_PIXELS = 24
ORDER = neopixel.GRBW
BRIGHTNESS = 0.75

# Detection
T = 155.5                        # gray threshold for the cream connector body
TRAY_ERODE = 15                # keeps the tray edge out of the result

# Connector filter (pixels)
LONG_RANGE = (140, 170)
SHORT_RANGE = (48, 92)
MIN_FILL = 0.75                # area / (long * short), 1.0 = perfect rectangle

# Pick point 
DPI_X = 10.597403195491156     # pixels per mm
X_OFFSET_MM = 15.7


# ----------------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------------
def draw_axis(img, p_, q_, colour, scale):
    p = list(p_)
    q = list(q_)

    angle = atan2(p[1] - q[1], p[0] - q[0])
    hypotenuse = sqrt((p[1] - q[1]) ** 2 + (p[0] - q[0]) ** 2)

    q[0] = p[0] - scale * hypotenuse * cos(angle)
    q[1] = p[1] - scale * hypotenuse * sin(angle)
    cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv.LINE_AA)

    p[0] = q[0] + 9 * cos(angle + pi / 4)
    p[1] = q[1] + 9 * sin(angle + pi / 4)
    cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv.LINE_AA)

    p[0] = q[0] + 9 * cos(angle - pi / 4)
    p[1] = q[1] + 9 * sin(angle - pi / 4)
    cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv.LINE_AA)


def get_orientation(pts, img):
    """PCA on a contour. Returns angle (rad), centre (x, y), eigenvectors."""
    data_pts = pts.reshape(-1, 2).astype(np.float64)

    mean, eigenvectors, eigenvalues = cv.PCACompute2(data_pts, np.empty((0)))
    cntr = (int(mean[0, 0]), int(mean[0, 1]))

    cv.circle(img, cntr, 3, (255, 0, 255), 2)
    p1 = (cntr[0] + 0.02 * eigenvectors[0, 0] * eigenvalues[0, 0],
          cntr[1] + 0.02 * eigenvectors[0, 1] * eigenvalues[0, 0])
    p2 = (cntr[0] - 0.02 * eigenvectors[1, 0] * eigenvalues[1, 0],
          cntr[1] - 0.02 * eigenvectors[1, 1] * eigenvalues[1, 0])
    draw_axis(img, cntr, p1, (0, 255, 0), 1)
    draw_axis(img, cntr, p2, (255, 255, 0), 5)

    angle = atan2(eigenvectors[0, 1], eigenvectors[0, 0])
    return angle, cntr, eigenvectors


def blob_stats(c):
    """Returns (long side, short side, fill) of the min-area rectangle."""
    _, (w, h), _ = cv.minAreaRect(c)
    long_s, short_s = max(w, h), min(w, h)
    fill = cv.contourArea(c) / (long_s * short_s) if short_s > 0 else 0.0
    return long_s, short_s, fill


def is_connector(c):
    long_s, short_s, fill = blob_stats(c)
    return (LONG_RANGE[0] < long_s < LONG_RANGE[1] and
            SHORT_RANGE[0] < short_s < SHORT_RANGE[1] and
            fill > MIN_FILL)


# ----------------------------------------------------------------------------
# Camera setup (still configuration at full sensor resolution)
# ----------------------------------------------------------------------------
picam2 = Picamera2()
config = picam2.create_still_configuration(
    main={"size": (3280, 2464), "format": "RGB888"},
    raw={"size": (3280, 2464)},
)
picam2.configure(config)

pixels = None
if USE_LED:
    pixels = neopixel.NeoPixel(PIXEL_PIN, NUM_PIXELS,
                               brightness=BRIGHTNESS, pixel_order=ORDER)

try:
    if USE_LED:
        pixels.fill((0, 0, 0, 255))     # R, G, B, W
        pixels.show()

    picam2.start()
    time.sleep(2)                       # let exposure / white balance settle

    img = picam2.capture_array("main")  # already BGR order for OpenCV with RGB888
    cv.imwrite("raw-png-PCA.png", img)

finally:
    picam2.stop()
    if USE_LED:
        pixels.fill((0, 0, 0, 0))
        pixels.show()

print("Image size (h, w, c):", img.shape)

# ----------------------------------------------------------------------------
# Detection
# ----------------------------------------------------------------------------
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
blur = cv.GaussianBlur(gray, (5, 5), 0)

# 1) Tray mask: the tray is the largest dark blob (inverted Otsu)
_, dark = cv.threshold(blur, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
cnts, _ = cv.findContours(dark, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
tray = max(cnts, key=cv.contourArea)
tray_mask = np.zeros_like(gray)
cv.drawContours(tray_mask, [tray], -1, 255, cv.FILLED)   # filled, so parts inside are kept
tray_mask = cv.erode(tray_mask, np.ones((TRAY_ERODE, TRAY_ERODE), np.uint8))

# --- Green PCB detection ---
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

# OpenCV hue runs 0-179, green is roughly 35-90. Tune S and V from your image.
GREEN_LO = (55, 60, 40)
GREEN_HI = (90, 255, 255)

green = cv.inRange(hsv, GREEN_LO, GREEN_HI)
green = cv.bitwise_and(green, tray_mask)      # only inside the tray
cv.imwrite("green_mask.png", green)

green_contours, _ = cv.findContours(green, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

MIN_BOARD_AREA = 25000                         # placeholder, set from the printout
for gc in green_contours:
    area = cv.contourArea(gc)
    if area < 500:                             # skip pixel dust in the printout
        continue
    print(f"green blob area={area:.0f}")

boards = [gc for gc in green_contours if cv.contourArea(gc) > MIN_BOARD_AREA]
cv.drawContours(img, boards, -1, (255, 0, 0), 3)   # blue in BGR

# Getting the center of the green PCB contour
MAX_LINK_DIST = 300      # px, connector centre to its board centroid (tune)
MIN_PROJ = 30            # px, below this the side is ambiguous

board_centres = []
for gc in boards:
    M = cv.moments(gc)
    board_centres.append(np.array([M["m10"] / M["m00"], M["m01"] / M["m00"]]))


# 2) Connector: plain fixed threshold, only inside the tray
_, bright = cv.threshold(blur, T, 255, cv.THRESH_BINARY)
mask = cv.bitwise_and(bright, tray_mask)
cv.imwrite("white_mask.png", mask)

contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

# 3) Debug: verdict for every blob
print(f"\nThreshold T = {T}, blobs found: {len(contours)}")
for c in contours:
    (cx, cy), _, _ = cv.minAreaRect(c)
    long_s, short_s, fill = blob_stats(c)
    if cv.contourArea(c) < 200:          # ignore pixel dust in the printout
        continue
    print(f"center=({cx:.0f},{cy:.0f}) long={long_s:.0f} short={short_s:.0f} "
          f"fill={fill:.2f} -> {is_connector(c)}")

# ----------------------------------------------------------------------------
# Orientation and pick point for accepted connectors
# ----------------------------------------------------------------------------
found = 0
for i, c in enumerate(contours):
    if not is_connector(c):
        continue
    found += 1

    # cv.drawContours(img, contours, i, (0, 0, 255), 2)
    angle, _, eigenvectors = get_orientation(c, img)
    M = cv.moments(c)
    cntr = np.array([M["m10"] / M["m00"], M["m01"] / M["m00"]])

    # cntr = np.array(cntr, dtype=float)
    eigenvectors = np.array(eigenvectors, dtype=float)
    
    # --- inside the connector loop, replacing the x_axis flip and pick_point lines ---
    x_axis = eigenvectors[1].copy()          # short axis, sign is arbitrary, no forcing

    if not board_centres:
        print("No green board found, skipping"); continue

    dists = [np.linalg.norm(bc - cntr) for bc in board_centres]
    j = int(np.argmin(dists))
    if dists[j] > MAX_LINK_DIST:
        print(f"Connector at {cntr}: no board within {MAX_LINK_DIST}px, skipping"); continue

    to_board = board_centres[j] - cntr
    proj = float(np.dot(to_board, x_axis))
    if abs(proj) < MIN_PROJ:
        print(f"Connector at {cntr}: side is ambiguous (proj={proj:.0f}), skipping"); continue
    
    print(f"link dist={dists[j]:.0f}px  proj={proj:.0f}px")

    u = x_axis if proj > 0 else -x_axis      # points from connector into the board
    pick_point = cntr + X_OFFSET_MM * DPI_X * u
    heading_deg = math.degrees(atan2(u[1], u[0]))   # full 360 deg heading

    cv.circle(img, tuple(board_centres[j].astype(int)), 6, (255, 0, 0), -1)  # board centroid
    cv.circle(img, tuple(pick_point.astype(int)), 5, (0, 255, 0), -1)
    cv.circle(img, tuple(pick_point.astype(int)), 5, (0, 255, 0), -1)

    print(f"\nConnector {found}: center={cntr}  pick={pick_point}") #   angle={tilt_deg:.2f} deg

print(f"\nConnectors accepted: {found} (expected 3)")
cv.imwrite("PCA-result.png", img)