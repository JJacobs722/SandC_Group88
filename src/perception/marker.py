"""Perception: turns a camera picture into the range and bearing of the dock marker.

What goes in:  an RGB camera image and the camera's K matrix
What comes out: (range_m, bearing_rad) measured from the ROBOT'S CENTRE,
                or None if the marker can't be seen.
                range is in metres, bearing in radians, positive = marker is to the LEFT.
"""

import math

import cv2
import numpy as np

from src.sim import config

# ---------- fixed numbers ----------
CAM_FORWARD = 0.07   # m: camera sits 7 cm in front of the robot's centre (robot file, front_cam pos)

# Measured in Task 4 (perception_noise.py). Update these if you re-measure.
RANGE_NOISE_K = 0.0099     # range spread grows with distance: std = RANGE_NOISE_K * range^2
BEARING_NOISE_STD = 0.0013  # rad: bearing spread stays about the same at every distance
RANGE_NOISE_FLOOR = 0.005   # m: never claim to be better than 5 mm
BEARING_NOISE_FLOOR = 0.001 # rad: never claim to be better than 0.001 rad

# The marker's 4 corners measured from its own centre (metres).
# Order must match what OpenCV gives back: top-left, top-right, bottom-right, bottom-left.
_half = config.MARKER_SIZE / 2
MARKER_CORNERS_3D = np.array([
    [-_half,  _half, 0],
    [ _half,  _half, 0],
    [ _half, -_half, 0],
    [-_half, -_half, 0],
], dtype=np.float32)

_dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
_default_rng = np.random.default_rng()


def _build_detector(refine_corners):
    params = cv2.aruco.DetectorParameters()
    if refine_corners:
        # Nudges each corner to the exact edge, to a fraction of a pixel. Big accuracy gain (Task 2).
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(_dictionary, params)


_detector = _build_detector(refine_corners=True)


def set_corner_refinement(on):
    """Turn corner refinement on or off. Only used to compare accuracy in Task 2."""
    global _detector
    _detector = _build_detector(refine_corners=on)


def detect_marker(image, K, rng=None):
    """Find the dock marker and return (range_m, bearing_rad) from the robot's centre, or None."""
    # 1. Find the marker's 4 corners in the picture (pixels)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    corners, ids, _ = _detector.detectMarkers(gray)
    if ids is None or config.MARKER_ID not in ids.flatten():
        return None
    index = list(ids.flatten()).index(config.MARKER_ID)
    pixel_corners = corners[index].reshape(4, 2).astype(np.float64)

    # 2. Simulated camera noise: wobble each corner by a little (set in config.py, 0 = off)
    noise = config.CAMERA_PIXEL_NOISE
    if noise > 0:
        rng = _default_rng if rng is None else rng
        pixel_corners = pixel_corners + rng.normal(0, noise, pixel_corners.shape)

    # 3. Work out where the marker is in 3D, relative to the camera
    ok, rvec, tvec = cv2.solvePnP(
        MARKER_CORNERS_3D, pixel_corners.astype(np.float32),
        np.asarray(K, dtype=np.float64), np.zeros(5),
        flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        return None
    cam_x, cam_y, cam_z = tvec.flatten()   # camera frame: x right, y down, z forward

    # 4. Change from the camera's point of view to the robot's
    robot_x = cam_z + CAM_FORWARD          # robot frame: x forward
    robot_y = -cam_x                       #              y left

    # 5. Turn that position into distance and direction
    range_m = math.hypot(robot_x, robot_y)
    bearing_rad = math.atan2(robot_y, robot_x)
    return range_m, bearing_rad


def measurement_covariance(range_m):
    """The R matrix for the EKF: how unsure a reading at this range is.

    Returns a 2x2 array: [[range variance, 0], [0, bearing variance]].
    """
    std_range = max(RANGE_NOISE_K * range_m ** 2, RANGE_NOISE_FLOOR)
    std_bearing = max(BEARING_NOISE_STD, BEARING_NOISE_FLOOR)
    return np.diag([std_range ** 2, std_bearing ** 2])