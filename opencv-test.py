import cv2
from picamera2 import Picamera2
picam2 = Picamera2()
picam2.preview_configuration.main.size = (1280,720)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()
while True:
    im= picam2.capture_array()
    cv2.imshow("Camera", im)
    if cv2.waitKey(1)==ord('q'):
        break
cv2.destroyAllWindows()






# import time
# import numpy as np
# from picamera2 import Picamera2, MappedArray

# import cv2
# picam2 = Picamera2()
# colour = (0, 255, 0)
# origin = (0, 30)
# font = cv2.FONT_HERSHEY_SIMPLEX
# scale = 1
# thickness = 2

# def apply_timestamp(request):
#   timestamp = time.strftime("%Y-%m-%d %X")
#   with MappedArray(request, "main") as m:
#     cv2.putText(m.array, timestamp, origin, font, scale, colour, thickness)

# picam2.pre_callback = apply_timestamp
# picam2.start(show_preview=True)
# time.sleep(5)

# picam2.start()


# while True:
#         frame = picam2.capture_array()



#         cv2.imshow('f', frame)
 
# Initialize video capture
# capture = cv2.VideoCapture(0)
# capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
# capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 576)
# capture.set(cv2.CAP_PROP_FPS, 30)  # Requesting 30 FPS from the camera
 
# # Background subtractor
# bg_subtractor = cv2.createBackgroundSubtractorMOG2()
 
# # Current mode
# mode = "normal"
 
# # FPS calculation
# prev_time = time.time()
 
# # Video writer initialization with explicit FPS
# save_fps = 15.0  # Adjust this based on your actual processing speed
# fourcc = cv2.VideoWriter_fourcc(*"XVID")
# out = cv2.VideoWriter("output.avi", fourcc, save_fps, (1024, 576))
 
# while True:
#     ret, frame = capture.read()
#     if not ret:
#         break
 
#     frame = cv2.flip(frame, 1)
#     display_frame = frame.copy()
 
#     if mode == "threshold":
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         _, display_frame = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
#         display_frame = cv2.cvtColor(display_frame, cv2.COLOR_GRAY2BGR)
 
#     elif mode == "edge":
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         edges = cv2.Canny(gray, 100, 200)
#         display_frame = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
 
#     elif mode == "bg_sub":
#         fg_mask = bg_subtractor.apply(frame)
#         display_frame = cv2.bitwise_and(frame, frame, mask=fg_mask)
 
#     elif mode == "contour":
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
#         contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#         display_frame = cv2.drawContours(frame.copy(), contours, -1, (0, 255, 0), 2)
 
#     # Calculate actual processing FPS
#     curr_time = time.time()
#     processing_fps = 1 / (curr_time - prev_time)
#     prev_time = curr_time
 
#     # Display actual processing FPS
#     cv2.putText(
#         display_frame, f"FPS: {int(processing_fps)} Mode: {mode}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
#     )
 
#     # Write frame to video
#     out.write(display_frame)
 
#     # Show video
#     cv2.imshow("Live Video", display_frame)
 
#     key = cv2.waitKey(1) & 0xFF
#     if key == ord("t"):
#         mode = "threshold"
#     elif key == ord("e"):
#         mode = "edge"
#     elif key == ord("b"):
#         mode = "bg_sub"
#     elif key == ord("c"):
#         mode = "contour"
#     elif key == ord("n"):
#         mode = "normal"
#     elif key == ord("q"):
#         break
 
# # Clean up
# capture.release()
# out.release()
# cv2.destroyAllWindows()