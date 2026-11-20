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

# The MATLAB calibration is only valid at exactly this resolution / sensor mode
IMG_W, IMG_H = 3280, 2464

# Size filter on the bounding box (w * h, in pixels at 3280x2464)
MIN_AREA = 3000
MAX_AREA = 6000

Z = 450                      # camera -> pick plane distance in mm (must match the setup!)
LINE_LENGTH_MM = 17

# Green probe: two small patches, one on each side of the connector along the line
PROBE_MM = 7.0               # distance from the connector centre to each patch centre
PATCH_HALF_MM = 1.0          # patch is a square of 2 * PATCH_HALF_MM on a side
GREEN_LO = np.array([55, 60, 40])     # OpenCV hue runs 0-179
GREEN_HI = np.array([90, 255, 255])
MIN_GREEN = 0.30             # the winning patch must be at least this green
MIN_DIFF = 0.25              # and clearly greener than the other patch

# ----------------------------------------------------------------------------
# Camera calibration parameters, obtained from MATLAB
# ----------------------------------------------------------------------------
fx = 2562.10035596914
fy = 2562.15173894501
cx = 1649.95172923836
cy = 1228.53049286571

k1 = 0.192807898722954
k2 = -0.373987768898079
p1 = 0.000666675204719012
p2 = -0.00118704988531241

K = np.array([[fx, 0, cx - 1],           # MATLAB counts pixels from 1, OpenCV from 0
              [0, fy, cy - 1],
              [0,  0,  1]])
dist = np.array([k1, k2, p1, p2, 0.0])   # OpenCV order: k1, k2, p1, p2, k3

# Pixels per mm on the pick plane. Pinhole model: one pixel covers Z / fx mm,
# so there are fx / Z pixels per mm (NOT Z / fx).
PX_PER_MM = fx / Z
print(f"Scale: {PX_PER_MM:.3f} px/mm  ({1 / PX_PER_MM:.4f} mm/px) at Z = {Z} mm")


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
# Hardware (full-resolution still configuration, same mode as the calibration images)
# ----------------------------------------------------------------------------
pixels = neopixel.NeoPixel(PIXEL_PIN, NUM_PIXELS, brightness=BRIGHTNESS, pixel_order=ORDER)

from picamera2 import Picamera2
picam2 = Picamera2()

config = picam2.create_still_configuration(
    main={"size": (IMG_W, IMG_H), "format": "RGB888"},
    raw={"size": (IMG_W, IMG_H)},
)
picam2.configure(config)

try:
    pixels.fill((0, 0, 0, 255))   # R, G, B, W
    pixels.show()

    picam2.start()
    time.sleep(2)                 # let exposure / white balance settle
    picam2.capture_file("test.png")

finally:
    picam2.stop()
    pixels.fill((0, 0, 0, 0))     # all channels off
    pixels.show()

img = cv2.imread("test.png")
if img.shape[1] != IMG_W or img.shape[0] != IMG_H:
    raise RuntimeError(f"Image is {img.shape[1]}x{img.shape[0]}, the calibration is for {IMG_W}x{IMG_H}")

# Remove lens distortion. Passing K as the new camera matrix keeps the scale (px/mm) unchanged.
img = cv2.undistort(img, K, dist, None, K)
cv2.imwrite("test-undistorted.png", img)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)      # from the clean image, before any drawing

# ----------------------------------------------------------------------------
# Threshold and contours
# ----------------------------------------------------------------------------
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Inverted Otsu: the tray is white, the connectors are holes inside it
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
cv2.imwrite("threshold-otsu-connector.png", thresh)

kernel = np.ones((5, 5), np.uint8)  # square kernel for morphology operations
# erosion = cv2.erode(thresh,kernel,iterations = 1) # thins out the contours
# opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

cnts = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_L1)  # change arg1, when changing morphlogy operation
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

print(f"\nContours passing the size filter: {len(part_cnts)} (expected 24)")

contour_filtered = img.copy()
cv2.drawContours(contour_filtered, part_cnts, -1, (0, 255, 0), 2)
cv2.imwrite("contours-filtered.png", contour_filtered)

# ----------------------------------------------------------------------------
# Centre, rotated rectangle, and a line pointing toward the PCB
# ----------------------------------------------------------------------------
probe_dist = PROBE_MM * PX_PER_MM
patch_half = int(round(PATCH_HALF_MM * PX_PER_MM))
pixel_length = LINE_LENGTH_MM * PX_PER_MM

ROW_TOL_MM = 3.0                       # two centres count as the same row if their y differs by less than this
row_tol = ROW_TOL_MM * PX_PER_MM       # same tolerance in pixels

parts = []
for k, c in enumerate(part_cnts, start=1):
    (x, y, w, h) = cv2.boundingRect(c)
    parts.append((x + w / 2, y + h / 2, c))

    # Dot in the centre of the bounding rectangle
    cx_part = x + w / 2
    cy_part = y + h / 2
    centre = np.array([cx_part, cy_part])
    cx_int, cy_int = int(cx_part), int(cy_part)
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

    print(f"\nPart {k}: centre=({cx_part:.0f},{cy_part:.0f})  green(+)={s_plus:.2f}  green(-)={s_minus:.2f}")

    # FINDING ITEMS IN SAME ROW
    parts.sort(key=lambda p: p[1])
    rows = []                              # each row is a list of (cx, cy, contour)
    for p in parts:
        if rows and abs(p[1] - np.mean([q[1] for q in rows[-1]])) < row_tol:
            rows[-1].append(p)
        else:
            rows.append([p])

    # Within each row, sort left to right
    for row in rows:
        row.sort(key=lambda p: p[0])

    print(f"\nFound {len(rows)} row(s)")
    for i, row in enumerate(rows, start=1):
        print(f"  Row {i}: {len(row)} part(s), mean y = {np.mean([p[1] for p in row]):.0f}")

    best = max(s_plus, s_minus)
    if best < MIN_GREEN or abs(s_plus - s_minus) < MIN_DIFF:
        print("  side is ambiguous, no line drawn (check the patches and GREEN_LO/GREEN_HI)")
        continue

    u = n if s_plus > s_minus else -n          # points from the connector toward the PCB
    heading_deg = math.degrees(math.atan2(u[1], u[0]))
    print(f"  PCB lies toward heading {heading_deg:.1f} deg (image coords, +x right, +y down)")

    x2 = int(cx_part + pixel_length * u[0])
    y2 = int(cy_part + pixel_length * u[1])
    cv2.line(contour_filtered, (cx_int, cy_int), (x2, y2), color=(0, 255, 0), thickness=3)


cv2.imwrite("rect-fil-contours.png", contour_filtered)
