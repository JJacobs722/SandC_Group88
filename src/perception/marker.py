"""Camera image -> range, bearing (and facing) of the dock marker."""

import math

import cv2
import numpy as np

from src.sim import config

# ---------- fixed numbers ----------
CAM_FORWARD = 0.07   # m: camera sits 7 cm in front of the robot's centre (robot file, front_cam pos)
CAM_LEFT = 0.0       # m: camera sits on the centre line in the sim
CAM_DIST = np.zeros(5)  # lens distortion (0 in sim)

# Measured with tools/perception_noise.py. Update these if you re-measure.
RANGE_NOISE_K = 0.0099     # range spread grows with distance: std = RANGE_NOISE_K * range^2
BEARING_NOISE_STD = 0.0013  # rad: bearing spread stays about the same at every distance
RANGE_NOISE_FLOOR = 0.005   # m: never claim to be better than 5 mm
BEARING_NOISE_FLOOR = 0.001 # rad: never claim to be better than 0.001 rad
# Facing noise, measured with tools/facing_noise.py. Gets much worse past 2 m.
FACING_NOISE_K = 0.045      # rad: std = FACING_NOISE_K * range^2
FACING_NOISE_FLOOR = 0.02   # rad: never claim better than about 1 deg
FACING_NOISE_CAP = 0.35     # rad: about 20 deg

# The marker's 4 corners measured from its own centre (metres).
# Order must match what OpenCV gives back: top-left, top-right, bottom-right, bottom-left.
def _corners_3d(size):
    h = size / 2
    return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float32)


MARKER_SIZE = config.MARKER_SIZE
MARKER_CORNERS_3D = _corners_3d(MARKER_SIZE)


def set_camera_setup(forward, left=0.0, dist=None, marker_size=None):
    """Camera position and lens, same call for sim and real robot."""
    global CAM_FORWARD, CAM_LEFT, CAM_DIST, MARKER_SIZE, MARKER_CORNERS_3D
    CAM_FORWARD, CAM_LEFT = float(forward), float(left)
    CAM_DIST = np.zeros(5) if dist is None else np.asarray(dist, dtype=np.float64).ravel()
    if marker_size is not None:
        MARKER_SIZE = float(marker_size)
        MARKER_CORNERS_3D = _corners_3d(MARKER_SIZE)

_dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
_default_rng = np.random.default_rng()


def _build_detector(refine_corners):
    params = cv2.aruco.DetectorParameters()
    if refine_corners:
        # Nudges each corner to the exact edge, to a fraction of a pixel. (see tools/perception_accuracy.py)
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(_dictionary, params)


_detector = _build_detector(refine_corners=True)


def set_corner_refinement(on):
    """Turn corner refinement on or off. Only used in tools/perception_accuracy.py."""
    global _detector
    _detector = _build_detector(refine_corners=on)


def _find_marker(image, K, rng=None):
    """Steps 1 to 3: find marker 0 and solve where it is relative to the camera. Returns (tvec, rvec) or None."""
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
        np.asarray(K, dtype=np.float64), CAM_DIST,
        flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        return None
    return tvec, rvec


def _range_bearing(tvec):
    """Steps 4 and 5: camera-frame position -> (range_m, bearing_rad) from the robot's centre."""
    cam_x, cam_y, cam_z = tvec.flatten()   # camera frame: x right, y down, z forward

    # 4. Change from the camera's point of view to the robot's
    robot_x = cam_z + CAM_FORWARD          # robot frame: x forward
    robot_y = -cam_x + CAM_LEFT            #              y left

    # 5. Turn that position into distance and direction
    range_m = math.hypot(robot_x, robot_y)
    bearing_rad = math.atan2(robot_y, robot_x)
    return range_m, bearing_rad


def detect_marker(image, K, rng=None):
    """Find the dock marker and return (range_m, bearing_rad) from the robot's centre, or None."""
    found = _find_marker(image, K, rng)
    if found is None:
        return None
    return _range_bearing(found[0])


def detect_marker_pose(image, K, rng=None):
    """Same as detect_marker, plus facing_rad (pi = marker facing straight at the robot)."""
    found = _find_marker(image, K, rng)
    if found is None:
        return None
    tvec, rvec = found
    range_m, bearing_rad = _range_bearing(tvec)

    # 6. The marker's z axis sticks straight out of its face. Rotate it into the robot frame.
    rot, _ = cv2.Rodrigues(rvec)
    out_x, out_y, out_z = rot[:, 2]        # camera frame: x right, y down, z forward
    facing_rad = math.atan2(-out_x, out_z) # robot frame: forward = out_z, left = -out_x
    return range_m, bearing_rad, facing_rad


def measurement_covariance(range_m):
    """The R matrix for the EKF: how unsure a reading at this range is.

    Returns a 2x2 array: [[range variance, 0], [0, bearing variance]].
    """
    std_range = max(RANGE_NOISE_K * range_m ** 2, RANGE_NOISE_FLOOR)
    std_bearing = max(BEARING_NOISE_STD, BEARING_NOISE_FLOOR)
    return np.diag([std_range ** 2, std_bearing ** 2])


def marker_covariance(range_m):
    """R for [range, bearing, facing]."""
    R2 = measurement_covariance(range_m)
    std_facing = min(max(FACING_NOISE_K * range_m ** 2, FACING_NOISE_FLOOR), FACING_NOISE_CAP)
    return np.diag([R2[0, 0], R2[1, 1], std_facing ** 2])
