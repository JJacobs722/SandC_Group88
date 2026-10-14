"""Task 5: calibrate the robot's camera with a checkerboard, like you would a real camera.

Uses its own small world (calibration_world.xml) with a floating checkerboard.
Moves the board to 30 random positions in front of the camera, photographs it,
then lets OpenCV work out the K matrix from the photos.

This tool loads MuJoCo directly because it needs to move the board.
That's fine for a calibration tool. The real pipeline still only uses Environment.
"""
from pathlib import Path

import cv2
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from src.sim import config
from src.sim.environment import Environment

ROOT = Path(__file__).resolve().parents[1]
WORLD = ROOT / "assets" / "robotis_tb3" / "calibration_world.xml"
INNER_CORNERS = (9, 6)    # corners where 4 squares meet, across and down
SQUARE_SIZE = 0.025       # m: each square is 2.5 cm
PHOTOS = 30

m = mujoco.MjModel.from_xml_path(str(WORLD))
d = mujoco.MjData(m)
renderer = mujoco.Renderer(m, config.CAMERA_HEIGHT, config.CAMERA_WIDTH)
rng = np.random.default_rng(0)

# Where each inner corner really is on the board, in metres (the board's own frame)
board_points = np.zeros((INNER_CORNERS[0] * INNER_CORNERS[1], 3), np.float32)
board_points[:, :2] = np.mgrid[0:INNER_CORNERS[0], 0:INNER_CORNERS[1]].T.reshape(-1, 2) * SQUARE_SIZE

all_board_points, all_image_points = [], []
for i in range(PHOTOS):
    # Random position and tilt in front of the camera
    distance = rng.uniform(0.45, 0.9)
    d.mocap_pos[0] = [0.07 + distance, rng.uniform(-0.12, 0.12), rng.uniform(0.04, 0.20)]
    turn = Rotation.from_euler("zyx", [rng.uniform(-35, 35), rng.uniform(-30, 30),
                                       rng.uniform(-15, 15)], degrees=True)
    qx, qy, qz, qw = turn.as_quat()
    d.mocap_quat[0] = [qw, qx, qy, qz]          # MuJoCo wants w first
    mujoco.mj_forward(m, d)

    renderer.update_scene(d, camera=config.CAMERA_NAME)
    image = renderer.render()
    if i == 0:
        cv2.imwrite("calibration_example.png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    found, corners = cv2.findChessboardCorners(gray, INNER_CORNERS)
    if found:
        corners = cv2.cornerSubPix(
            gray, corners, (5, 5), (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        all_board_points.append(board_points)
        all_image_points.append(corners)

print(f"Board found in {len(all_image_points)} of {PHOTOS} photos")

rms, K, dist, _, _ = cv2.calibrateCamera(
    all_board_points, all_image_points, (config.CAMERA_WIDTH, config.CAMERA_HEIGHT), None, None,
    flags=cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3)

np.set_printoptions(precision=1, suppress=True)
print(f"\nReprojection error (rms): {rms:.3f} pixels   (under 0.2 is good)")
print("\nK from calibration:")
print(K)
print("\nK from the formula in environment.py:")
print(Environment().get_camera_intrinsics())
print(f"\nLens distortion (k1, k2): {dist.ravel()[0]:.3f}, {dist.ravel()[1]:.3f}   (near 0 = no distortion)")
np.savez("camera_calibration.npz", K=K, dist=dist, rms=rms)
print("Saved camera_calibration.npz and calibration_example.png")