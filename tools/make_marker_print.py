"""Make the ArUco marker to PRINT for the real robot (ID 0, DICT_4X4_50).

Not flipped: the flip in make_marker.py is only because MuJoCo mirrors the picture on the box.
Print it so the black square is MARKER_SIZE wide (20 cm by default) and measure it after printing.
  python -m tools.make_marker_print
"""
import cv2

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
marker = cv2.aruco.generateImageMarker(dictionary, 0, 1200)                              # high resolution for printing
marker = cv2.copyMakeBorder(marker, 150, 150, 150, 150, cv2.BORDER_CONSTANT, value=255)  # white border is needed
cv2.imwrite("aruco_0_print.png", marker)
print("Saved aruco_0_print.png. Print it so the BLACK square is 20 cm wide, then measure it and set MARKER_SIZE in src/hw/hw_config.py.")
