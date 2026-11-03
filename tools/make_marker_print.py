"""ArUco marker ID 0 to print for the real robot (not mirrored). Set MARKER_SIZE to the measured black square.
  python -m tools.make_marker_print
"""
import cv2

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
marker = cv2.aruco.generateImageMarker(dictionary, 0, 1200)                              # high resolution for printing
marker = cv2.copyMakeBorder(marker, 150, 150, 150, 150, cv2.BORDER_CONSTANT, value=255)  # white border is needed
cv2.imwrite("aruco_0_print.png", marker)
print("Saved aruco_0_print.png. Print it so the BLACK square is 20 cm wide, then measure it and set MARKER_SIZE in src/hw/hw_config.py.")
