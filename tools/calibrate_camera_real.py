"""Calibrate the real camera with a checkerboard (printed or on a screen). Saves camera_calibration_real.npz.
  python -m tools.calibrate_camera_real   (SPACE keep photo, C calibrate, Q quit)
"""
import cv2
import numpy as np

from src.hw.real_robot import RealRobot

INNER_CORNERS = (9, 6)    # corners where 4 squares meet (the board is 10 x 7 squares)
SQUARE_SIZE = 0.018       # m: MEASURE your printed square and change this
OUT_FILE = "camera_calibration_real.npz"

board = np.zeros((INNER_CORNERS[0] * INNER_CORNERS[1], 3), np.float32)
board[:, :2] = np.mgrid[0:INNER_CORNERS[0], 0:INNER_CORNERS[1]].T.reshape(-1, 2) * SQUARE_SIZE

env = RealRobot()
board_pts, image_pts, size = [], [], None
try:
    while True:
        rgb = env.get_camera_image()
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        size = gray.shape[::-1]
        found, corners = cv2.findChessboardCorners(gray, INNER_CORNERS)
        shown = bgr.copy()
        if found:
            corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1),
                                       (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            cv2.drawChessboardCorners(shown, INNER_CORNERS, corners, found)
        cv2.putText(shown, f"kept {len(image_pts)}  SPACE keep  C calibrate  Q quit", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow("camera calibration", shown)
        key = cv2.waitKey(30) & 0xFF
        if key == ord(" ") and found:
            board_pts.append(board)
            image_pts.append(corners)
            print(f"kept photo {len(image_pts)}")
        elif key == ord("c") and len(image_pts) >= 10:
            break
        elif key == ord("q"):
            raise SystemExit("quit, nothing saved")
finally:
    cv2.destroyAllWindows()
    env.close()

rms, K, dist, _, _ = cv2.calibrateCamera(board_pts, image_pts, size, None, None)
np.set_printoptions(precision=2, suppress=True)
print(f"\nreprojection error (rms): {rms:.3f} pixels  (under 0.5 is good for a real camera)")
print("K =\n", K)
print("distortion (k1, k2, p1, p2, k3) =", dist.ravel())
np.savez(OUT_FILE, K=K, dist=dist, rms=rms, size=np.array(size))
print(f"saved {OUT_FILE}. Compare with the sim K (f = 415.7) on the project page.")
