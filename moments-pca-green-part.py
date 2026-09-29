import os
os.environ["QT_QPA_PLATFORM"] = "wayland"

import time
import math

import cv2 as cv
import numpy as np
import board
import neopixel
from picamera2 import Picamera2

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
USE_LED = True                 # keep identical to the lighting used while tuning

PIXEL_PIN = board.D18
NUM_PIXELS = 24
ORDER = neopixel.GRBW
BRIGHTNESS = 1.0


# Connector detection
T = 150                        # gray threshold for the cream connector body
TRAY_ERODE = 15                # keeps the tray edge out of the result
LONG_RANGE = (140, 180)
SHORT_RANGE = (48, 92)
MIN_FILL = 0.68                # area / (long * short), 1.0 = perfect rectangle

# Green PCB detection (OpenCV hue runs 0-179)
GREEN_LO = (55, 60, 40)
GREEN_HI = (90, 255, 255)
MIN_BOARD_AREA = 25000

# Connector <-> board pairing (your printout: link dist ~130 px, proj ~130 px)
MAX_LINK_DIST = 200            # px
MIN_PROJ = 45                  # px

# Pick point
PX_PER_MM = 10.597403195491156
PICK_OFFSET_MM = 5.0           # depth of the pick point, measured from the PCB centre along u
                               # +  = away from the connector, - = toward the connector
                               # (sideways position is taken from the connector centre)


# ----------------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------------
def ipt(p):
    return (int(round(p[0])), int(round(p[1])))


def draw_dir(img, p, d, length, colour, thick=2):
    p = np.asarray(p, dtype=float)
    cv.line(img, ipt(p), ipt(p + d * length), colour, thick, cv.LINE_AA)


def centroid(c):
    """Area centroid of a contour, or None if the contour is degenerate."""
    M = cv.moments(c)
    if M["m00"] == 0:
        return None
    return np.array([M["m10"] / M["m00"], M["m01"] / M["m00"]])


def moment_frame(c):
    """Centre and axes of a contour, all from image moments.
    Returns (centre, long_axis, short_axis) as float arrays, or None."""
    M = cv.moments(c)
    if M["m00"] == 0:
        return None
    centre = np.array([M["m10"] / M["m00"], M["m01"] / M["m00"]])
    theta = 0.5 * math.atan2(2 * M["mu11"], M["mu20"] - M["mu02"])   # long-axis angle
    long_axis = np.array([math.cos(theta), math.sin(theta)])
    short_axis = np.array([-long_axis[1], long_axis[0]])
    return centre, long_axis, short_axis


def aligned_box(contour, origin, u, v):
    """Bounding rectangle of a contour in the (u, v) frame. Its sides are parallel to
    u and v, so it is never rotated relative to the connector. Returns 4 int corners."""
    pts = contour.reshape(-1, 2).astype(float) - origin
    pu, pv = pts @ u, pts @ v
    corners = [origin + a * u + b * v
               for a, b in ((pu.min(), pv.min()), (pu.max(), pv.min()),
                            (pu.max(), pv.max()), (pu.min(), pv.max()))]
    return np.array([ipt(p) for p in corners], dtype=np.int32)


def angle_deg(v):
    return math.degrees(math.atan2(v[1], v[0]))


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

    img = picam2.capture_array("main")  # BGR order for OpenCV with RGB888
    cv.imwrite("raw-png-PCA.png", img)

finally:
    picam2.stop()
    if USE_LED:
        pixels.fill((0, 0, 0, 0))
        pixels.show()

print("Image size (h, w, c):", img.shape)

# ----------------------------------------------------------------------------
# Masks (computed before anything is drawn on img)
# ----------------------------------------------------------------------------
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
blur = cv.GaussianBlur(gray, (5, 5), 0)
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

# Tray mask: the tray is the largest dark blob (inverted Otsu)
_, dark = cv.threshold(blur, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
cnts, _ = cv.findContours(dark, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
tray = max(cnts, key=cv.contourArea)
tray_mask = np.zeros_like(gray)
cv.drawContours(tray_mask, [tray], -1, 255, cv.FILLED)
tray_mask = cv.erode(tray_mask, np.ones((TRAY_ERODE, TRAY_ERODE), np.uint8))

# Green PCBs
green = cv.bitwise_and(cv.inRange(hsv, GREEN_LO, GREEN_HI), tray_mask)
cv.imwrite("green_mask.png", green)
green_contours, _ = cv.findContours(green, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

for gc in green_contours:
    a = cv.contourArea(gc)
    if a >= 500:
        print(f"green blob area={a:.0f}")

boards = []
board_centres = []
for gc in green_contours:
    if cv.contourArea(gc) <= MIN_BOARD_AREA:
        continue
    bc = centroid(gc)
    if bc is None:
        continue
    boards.append(gc)
    board_centres.append(bc)

# Connectors
_, bright = cv.threshold(blur, T, 255, cv.THRESH_BINARY)
mask = cv.bitwise_and(bright, tray_mask)
cv.imwrite("white_mask.png", mask)
contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

print(f"\nThreshold T = {T}, blobs found: {len(contours)}")
for c in contours:
    if cv.contourArea(c) < 200:          # ignore pixel dust in the printout
        continue
    (cx, cy), _, _ = cv.minAreaRect(c)
    long_s, short_s, fill = blob_stats(c)
    print(f"center=({cx:.0f},{cy:.0f}) long={long_s:.0f} short={short_s:.0f} "
          f"fill={fill:.2f} -> {is_connector(c)}")

# ----------------------------------------------------------------------------
# Pick point for every accepted connector
# ----------------------------------------------------------------------------
cv.drawContours(img, boards, -1, (255, 0, 0), 3)       # green PCBs, blue outline

found = 0
for c in contours:
    if not is_connector(c):
        continue

    frame = moment_frame(c)
    if frame is None:
        continue
    cntr, long_ax, short_ax = frame

    if not board_centres:
        print("No green board found, skipping")
        continue

    # Pair with the nearest board
    dists = [np.linalg.norm(bc - cntr) for bc in board_centres]
    j = int(np.argmin(dists))
    if dists[j] > MAX_LINK_DIST:
        print(f"Connector at {cntr.round(0)}: no board within {MAX_LINK_DIST}px, skipping")
        continue

    # Which way along the connector's short axis does the board lie?
    board_c = board_centres[j]
    to_board = board_c - cntr
    proj = float(np.dot(to_board, short_ax))
    if abs(proj) < MIN_PROJ:
        print(f"Connector at {cntr.round(0)}: side is ambiguous (proj={proj:.0f}), skipping")
        continue

    # The PCB frame IS the connector frame
    u = short_ax if proj > 0 else -short_ax        # connector -> board, into the PCB
    v = np.array([-u[1], u[0]])                    # along the connector's long side

    # Pick point: depth from the PCB centre along u, sideways position from the connector
    # centre, so it lies opposite the middle of the connector's long side
    lateral = float(np.dot(cntr - board_c, v))
    pick_point = board_c + PICK_OFFSET_MM * PX_PER_MM * u + lateral * v

    inside = cv.pointPolygonTest(boards[j], (float(pick_point[0]), float(pick_point[1])), True)
    if inside < 0:
        print(f"Connector at {cntr.round(0)}: pick point is {-inside:.0f}px OFF the board, skipping")
        continue

    found += 1

    # Drawing
    cv.drawContours(img, [c], -1, (0, 0, 255), 2)                                   # connector
    cv.circle(img, ipt(cntr), 5, (255, 0, 255), -1)                                 # connector centre
    draw_dir(img, cntr, long_ax, 60, (0, 255, 0), 2)                                # connector long axis
    draw_dir(img, cntr, u, 60, (255, 255, 0), 2)                                    # into-the-board axis
    box = aligned_box(boards[j], board_c, u, v)
    cv.polylines(img, [box], True, (0, 255, 255), 2)                                # PCB box, aligned to connector
    cv.circle(img, ipt(board_c), 6, (255, 0, 0), -1)                                # PCB centre
    cv.line(img, ipt(board_c), ipt(pick_point), (0, 255, 0), 1, cv.LINE_AA)
    cv.circle(img, ipt(pick_point), 7, (0, 255, 0), -1)                             # pick point

    # Sanity check: the connector->PCB-centre line differs from u only by atan(lateral / depth)
    u_line = to_board / np.linalg.norm(to_board)
    print(f"\nConnector {found}: centre={cntr.round(1)}  board centre={board_c.round(1)}")
    print(f"  link dist={dists[j]:.0f}px ({dists[j] / PX_PER_MM:.1f}mm)  proj={proj:.0f}px")
    print(f"  heading (connector axis) = {angle_deg(u):.2f} deg   centre-to-centre = {angle_deg(u_line):.2f} deg")
    print(f"  lateral shift = {lateral:+.1f}px ({lateral / PX_PER_MM:+.2f}mm)")
    print(f"  pick={pick_point.round(1)}  margin to board edge={inside:.0f}px")

print(f"\nConnectors accepted: {found} (expected 3)")
cv.imwrite("PCA-result.png", img)