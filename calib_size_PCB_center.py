import os
os.environ["QT_QPA_PLATFORM"] = "wayland"

import cv2 as cv2
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

# Black tray (outer size). A part only counts if its contour lies inside the tray contour in the tree.
TRAY_LONG_MM, TRAY_SHORT_MM = 185.6, 146.6
TRAY_TOL = 0.10              # allowed deviation of the measured tray size

# Green PCB detection (HSV is computed from the undistorted image)
GREEN_LO = np.array([55, 40, 20])     # OpenCV hue runs 0-179. Loosen S/V if boards fragment in dark areas
GREEN_HI = np.array([90, 255, 255])
USE_TRAY_MASK = True                  # ignore green outside the tray contour

BOARD_AREA_MM2 = 265.0                # expected green area of ONE board (measured earlier, tune it)
BOARD_AREA_MIN_FRAC = 0.50            # blob smaller than this fraction  -> fragment, rejected
BOARD_AREA_MAX_FRAC = 1.40            # blob larger than this fraction   -> merged boards, rejected
BOARD_CLOSE_MM = 1.0                  # closing kernel, bridges small holes inside one board (keep < gap between boards)

# Connector -> board pairing
BOARD_DIST_MM = 12.0                  # expected distance connector centre -> its board's green centroid (tune it!)
BOARD_DIST_TOL_MM = 4.0               # allowed deviation from that distance
MAX_SIDE_MM = 5.0                     # max sideways offset of the board centroid from the connector axis
MIN_PROJ_MM = 2.0                     # min projection on the axis, so the sign cannot flip at random
MAX_HEADING_DIFF_DEG = 20.0           # connector-to-board heading vs. line heading, warning only

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

# Pixels per mm on the pick plane. Pinhole model: fx / Z pixels per mm.
PX_PER_MM = fx / Z
print(f"Scale: {PX_PER_MM:.3f} px/mm  ({1 / PX_PER_MM:.4f} mm/px) at Z = {Z} mm")


# ----------------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------------
def find_tray(cnts):
    """Index of the contour whose rotated-rectangle size matches the tray (largest one if several).
    Returns None if nothing matches."""
    best_idx, best_area = None, 0.0
    largest = None
    for i, c in enumerate(cnts):
        _, (w, h), _ = cv2.minAreaRect(c)
        long_mm, short_mm = max(w, h) / PX_PER_MM, min(w, h) / PX_PER_MM
        area = cv2.contourArea(c)
        if largest is None or area > largest[0]:
            largest = (area, long_mm, short_mm)
        if (abs(long_mm - TRAY_LONG_MM) < TRAY_TOL * TRAY_LONG_MM and
                abs(short_mm - TRAY_SHORT_MM) < TRAY_TOL * TRAY_SHORT_MM and
                area > best_area):
            best_idx, best_area = i, area
    if best_idx is None and largest is not None:
        print(f"Tray not found. Largest contour is {largest[1]:.0f} x {largest[2]:.0f} mm, "
              f"expected {TRAY_LONG_MM} x {TRAY_SHORT_MM} mm")
    return best_idx


def tree_depth_below(idx, root_idx, hier):
    """How many levels below root_idx the contour idx sits in the contour tree
    (1 = direct child, 2 = grandchild, ...). None if idx is not inside root_idx."""
    depth = 0
    parent = hier[idx][3]                  # hierarchy row = [next, previous, first child, parent]
    while parent != -1:
        depth += 1
        if parent == root_idx:
            return depth
        parent = hier[parent][3]
    return None


def find_boards(hsv_img, tray_contour=None):
    """Detect the green PCBs. Returns a list of dicts with the centroid and area (px),
    plus the cleaned mask for inspection. Merged blobs and fragments are reported and rejected."""
    mask = cv2.inRange(hsv_img, GREEN_LO, GREEN_HI)

    if tray_contour is not None:
        tray_mask = np.zeros(mask.shape, np.uint8)
        cv2.drawContours(tray_mask, [tray_contour], -1, 255, -1)
        mask = cv2.bitwise_and(mask, tray_mask)

    # remove dust, then close small holes inside one board
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    k = int(round(BOARD_CLOSE_MM * PX_PER_MM)) | 1          # odd kernel size
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))

    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)

    a_exp = BOARD_AREA_MM2 * PX_PER_MM ** 2                 # scale-independent: defined in mm^2
    a_min, a_max = BOARD_AREA_MIN_FRAC * a_exp, BOARD_AREA_MAX_FRAC * a_exp
    print(f"\nExpected board area: {a_exp:.0f} px  (accepted {a_min:.0f} .. {a_max:.0f})")

    boards = []
    for i in range(1, n):                                    # label 0 is background
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 0.15 * a_exp:                              # pixel dust, stay quiet
            continue
        cxb, cyb = cents[i]
        if area > a_max:
            print(f"  green blob at ({cxb:.0f},{cyb:.0f}) area={area} too large -> boards merged? rejected")
            continue
        if area < a_min:
            print(f"  green blob at ({cxb:.0f},{cyb:.0f}) area={area} too small -> fragment? rejected "
                  f"(check green-mask.png, loosen GREEN_LO S/V)")
            continue
        boards.append({"centroid": np.array([cxb, cyb]), "area": int(area)})

    print(f"Boards found: {len(boards)}")
    return boards, mask


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
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

cnts, hierarchy = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_L1)  # change arg1, when changing morphlogy operation
hier = hierarchy[0] if hierarchy is not None else []

contour_unfiltered = img.copy()
cv2.drawContours(contour_unfiltered, cnts, -1, (0, 255, 0), 2)
cv2.imwrite("contours-unfiltered.png", contour_unfiltered)

# ----------------------------------------------------------------------------
# Size filter (absolute window, no average)
# ----------------------------------------------------------------------------
tray_idx = find_tray(cnts)
if tray_idx is not None:
    _, (tw, th), _ = cv2.minAreaRect(cnts[tray_idx])
    print(f"Tray found: contour {tray_idx}, {max(tw, th) / PX_PER_MM:.1f} x {min(tw, th) / PX_PER_MM:.1f} mm")

part_cnts = []
for i, c in enumerate(cnts):
    (x, y, w, h) = cv2.boundingRect(c)
    area = w * h
    if area >= 500:                       # print only meaningful blobs, not pixel dust
        print(f"bbox area={area}  at ({x},{y}) size {w}x{h}")
    if not (MIN_AREA < area < MAX_AREA):
        continue

    depth = tree_depth_below(i, tray_idx, hier) if tray_idx is not None else None
    if depth is None:
        print(f"  -> rejected: not inside the tray (centre {x + w // 2},{y + h // 2})")
        continue
    print(f"  -> inside the tray, tree depth {depth}")
    part_cnts.append(c)

print(f"\nContours passing the size and tray filters: {len(part_cnts)} (expected 3)")

contour_filtered = img.copy()
cv2.drawContours(contour_filtered, part_cnts, -1, (0, 255, 0), 2)
if tray_idx is not None:
    cv2.drawContours(contour_filtered, cnts, tray_idx, (255, 0, 0), 3)     # tray outline, blue
cv2.imwrite("contours-filtered.png", contour_filtered)

# ----------------------------------------------------------------------------
# Detect the green PCBs (centroids)
# ----------------------------------------------------------------------------
tray_cnt = cnts[tray_idx] if tray_idx is not None and USE_TRAY_MASK else None
boards, green_mask = find_boards(hsv, tray_cnt)
cv2.imwrite("green-mask.png", green_mask)

for b in boards:
    bx, by = int(b["centroid"][0]), int(b["centroid"][1])
    cv2.drawMarker(contour_filtered, (bx, by), (255, 0, 255), cv2.MARKER_CROSS, 30, 3)   # magenta cross = board centroid

# ----------------------------------------------------------------------------
# Connector geometry: centre, rotated rectangle, long / short axis
# ----------------------------------------------------------------------------
parts = []
for k, c in enumerate(part_cnts, start=1):
    (x, y, w, h) = cv2.boundingRect(c)
    centre = np.array([x + w / 2, y + h / 2])
    cv2.circle(contour_filtered, (int(centre[0]), int(centre[1])), 5, (0, 0, 255), -1)

    rotrect = cv2.minAreaRect(c)
    box = np.int32(cv2.boxPoints(rotrect))
    cv2.drawContours(contour_filtered, [box], 0, (0, 0, 255), 2)

    # Longer edge of the rectangle = connector's long axis
    edge1 = box[1] - box[0]
    edge2 = box[2] - box[1]
    long_edge = edge1 if np.linalg.norm(edge1) > np.linalg.norm(edge2) else edge2
    t = long_edge / np.linalg.norm(long_edge)              # unit vector along the long side
    n = np.array([-t[1], t[0]])                            # unit vector along the short side (perpendicular), sign still unknown

    parts.append({"id": k, "contour": c, "centre": centre, "t": t, "n": n, "board": None})

# ----------------------------------------------------------------------------
# Pair every connector with ITS board (one to one)
# ----------------------------------------------------------------------------
# For every connector/board combination:
#   v    = board centroid - connector centre
#   proj = v . n   (distance along the short axis, its sign gives the direction of the line)
#   side = v . t   (sideways offset along the long axis, should be small)
#   dist = |v|     (should be about BOARD_DIST_MM)
# A combination is a candidate only if all three are plausible. Then the best candidates
# are taken first, and a board or a connector can be used only once.
cands = []
for pi, p in enumerate(parts):
    for bi, b in enumerate(boards):
        v = b["centroid"] - p["centre"]
        proj, side, d = float(v @ p["n"]), float(v @ p["t"]), float(np.linalg.norm(v))
        d_mm, side_mm, proj_mm = d / PX_PER_MM, side / PX_PER_MM, proj / PX_PER_MM
        if (abs(d_mm - BOARD_DIST_MM) <= BOARD_DIST_TOL_MM and
                abs(side_mm) <= MAX_SIDE_MM and
                abs(proj_mm) >= MIN_PROJ_MM):
            cost = abs(d_mm - BOARD_DIST_MM) + abs(side_mm)
            cands.append((cost, pi, bi, proj, side_mm, d_mm))

cands.sort(key=lambda c: c[0])
used_parts, used_boards = set(), set()
for cost, pi, bi, proj, side_mm, d_mm in cands:
    if pi in used_parts or bi in used_boards:
        continue
    used_parts.add(pi)
    used_boards.add(bi)
    parts[pi]["board"] = {"index": bi, "proj": proj, "side_mm": side_mm, "dist_mm": d_mm, "cost": cost}

# Report ambiguity: a part with more than one plausible board
for pi, p in enumerate(parts):
    n_cands = sum(1 for c in cands if c[1] == pi)
    if n_cands > 1:
        print(f"Part {p['id']}: {n_cands} plausible boards, took the best fit. Check the picture.")

for bi in range(len(boards)):
    if bi not in used_boards:
        bc = boards[bi]["centroid"]
        print(f"Board at ({bc[0]:.0f},{bc[1]:.0f}) has no connector assigned")

# ----------------------------------------------------------------------------
# Draw the line: along the connector's short axis, sign from the board
# ----------------------------------------------------------------------------
pixel_length = LINE_LENGTH_MM * PX_PER_MM

for p in parts:
    cxp, cyp = p["centre"]
    print(f"\nPart {p['id']}: centre=({cxp:.0f},{cyp:.0f})")

    m = p["board"]
    if m is None:
        print("  no matching board found, no line drawn (check green-mask.png, BOARD_DIST_MM and tolerances)")
        continue

    sign = 1.0 if m["proj"] > 0 else -1.0
    u = sign * p["n"]                                      # points from the connector toward the PCB
    heading_deg = math.degrees(math.atan2(u[1], u[0]))

    # Cross-check: heading of the connector -> board vector vs. the line heading
    v = boards[m["index"]]["centroid"] - p["centre"]
    v_heading = math.degrees(math.atan2(v[1], v[0]))
    diff = abs((v_heading - heading_deg + 180) % 360 - 180)

    print(f"  board dist={m['dist_mm']:.1f} mm, side offset={m['side_mm']:+.1f} mm, area={boards[m['index']]['area']} px")
    print(f"  PCB lies toward heading {heading_deg:.1f} deg (image coords, +x right, +y down), "
          f"heading to board centroid {v_heading:.1f} deg (diff {diff:.1f})")
    if diff > MAX_HEADING_DIFF_DEG:
        print("  WARNING: heading to the board differs a lot from the connector axis, check this pair")

    x2 = int(cxp + pixel_length * u[0])
    y2 = int(cyp + pixel_length * u[1])
    cv2.line(contour_filtered, (int(cxp), int(cyp)), (x2, y2), color=(0, 255, 0), thickness=3)

cv2.imwrite("rect-fil-contours.png", contour_filtered)