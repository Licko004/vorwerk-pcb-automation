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

# Size filter on the bounding box (w * h, in pixels at 1920x1080)
MIN_AREA = 3500
MAX_AREA = 8000

DPI = 5.795367506260126 #6.279251163812498      # pixels per mm (must match the current camera height!)
LINE_LENGTH_MM = 17

# Green probe: two small patches, one on each side of the connector along the line
PROBE_MM = 9.0               # distance from the connector centre to each patch centre
PATCH_HALF_MM = 2.0          # patch is a square of 2 * PATCH_HALF_MM on a side
GREEN_LO = np.array([55, 60, 40])     # OpenCV hue runs 0-179
GREEN_HI = np.array([90, 255, 255])
MIN_GREEN = 0.30             # the winning patch must be at least this green
MIN_DIFF = 0.25              # and clearly greener than the other patch


# ----------------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------------
def green_fraction(hsv_img, centre, half):
    """Fraction of green pixels in a square patch around centre (clamped to the image).
    Returns (fraction, (x0, y0, x1, y1))."""
    h, w = hsv_img.shape[:2]
    x0, x1 = max(int(centre[0]) - half, 0), min(int(centre[0]) + half + 1, w)
    y0, y1 = max(int(centre[1]) - half, 0), min(int(centre[1]) + half + 1, h)
    if x1 <= x0 or y1 <= y0:
        return 0.0, (x0, y0, x1, y1)
    patch = hsv_img[y0:y1, x0:x1]
    mask = cv2.inRange(patch, GREEN_LO, GREEN_HI)
    return float(np.count_nonzero(mask)) / mask.size, (x0, y0, x1, y1)


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
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)      # from the clean image, before any drawing

# ----------------------------------------------------------------------------
# Threshold and contours
# ----------------------------------------------------------------------------
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Inverted Otsu: the tray is white, the connectors are holes inside it
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
cv2.imwrite("threshold-otsu-connector.png", thresh)

cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_L1)
cnts = imutils.grab_contours(cnts)

contour_unfiltered = img.copy()
cv2.drawContours(contour_unfiltered, cnts, -1, (0, 255, 0), 2)
cv2.imwrite("contours-unfiltered.png", contour_unfiltered)

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
# Centre, rotated rectangle, and a line pointing toward the PCB
# ----------------------------------------------------------------------------
probe_dist = PROBE_MM * DPI
patch_half = int(round(PATCH_HALF_MM * DPI))
pixel_length = LINE_LENGTH_MM * DPI

for k, c in enumerate(part_cnts, start=1):
    (x, y, w, h) = cv2.boundingRect(c)

    # Dot in the centre of the bounding rectangle
    cx = x + w / 2
    cy = y + h / 2
    centre = np.array([cx, cy])
    cx_int, cy_int = int(cx), int(cy)
    cv2.circle(contour_filtered, (cx_int, cy_int), 5, (0, 0, 255), -1)

    rotrect = cv2.minAreaRect(c)
    box = np.int32(cv2.boxPoints(rotrect))
    cv2.drawContours(contour_filtered, [box], 0, (0, 0, 255), 2)

    # Find the longer edge of the rectangle to compute the angle
    edge1 = box[1] - box[0]
    edge2 = box[2] - box[1]
    long_edge = edge1 if np.linalg.norm(edge1) > np.linalg.norm(edge2) else edge2

    long_edge_angle = math.degrees(math.atan2(long_edge[1], long_edge[0]))
    line_angle = long_edge_angle + 90          # perpendicular to the long edge (sign still unknown)
    n = np.array([math.cos(math.radians(line_angle)), math.sin(math.radians(line_angle))])

    # Probe both directions for green
    p_plus = centre + probe_dist * n
    p_minus = centre - probe_dist * n
    s_plus, r_plus = green_fraction(hsv, p_plus, patch_half)
    s_minus, r_minus = green_fraction(hsv, p_minus, patch_half)

    for r in (r_plus, r_minus):                # show where the patches are, for tuning
        cv2.rectangle(contour_filtered, (r[0], r[1]), (r[2], r[3]), (255, 255, 0), 1)

    print(f"\nPart {k}: centre=({cx:.0f},{cy:.0f})  green(+)={s_plus:.2f}  green(-)={s_minus:.2f}")

    best = max(s_plus, s_minus)
    if best < MIN_GREEN or abs(s_plus - s_minus) < MIN_DIFF:
        print("  side is ambiguous, no line drawn (check the patches and GREEN_LO/GREEN_HI)")
        continue

    u = n if s_plus > s_minus else -n          # points from the connector toward the PCB
    heading_deg = math.degrees(math.atan2(u[1], u[0]))
    print(f"  PCB lies toward heading {heading_deg:.1f} deg (image coords, +x right, +y down)")

    x2 = int(cx + pixel_length * u[0])
    y2 = int(cy + pixel_length * u[1])
    cv2.line(contour_filtered, (cx_int, cy_int), (x2, y2), color=(0, 255, 0), thickness=3)

cv2.imwrite("rect-fil-contours.png", contour_filtered)