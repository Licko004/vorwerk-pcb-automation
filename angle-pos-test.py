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
picam2.capture_file("test.jpg")
