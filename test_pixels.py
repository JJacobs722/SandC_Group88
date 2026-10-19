import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import cv2
import numpy as np
from src.sim.environment import Environment
from src.sim import config
from src.perception import marker

env = Environment(seed=0)
K = env.get_camera_intrinsics()
print("K =\n", np.round(K, 1), "\nimage:", config.CAMERA_WIDTH, "x", config.CAMERA_HEIGHT)

env.reset(1.4, 0.0, 0.0)                      # 1.04 m from the dock, facing +x
img = env.get_camera_image()
print("rendered image shape:", img.shape)
corners, ids, _ = marker._detector.detectMarkers(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
print("ids:", None if ids is None else ids.flatten())

if ids is not None:
    c = corners[0].reshape(4, 2)
    u, v = c.mean(axis=0)
    print("corners (px):\n", np.round(c, 1))
    print(f"marker centre: u={u:.0f} v={v:.0f} | principal point: {K[0,2]:.0f}, {K[1,2]:.0f}")
    print(f"angle off axis from PIXELS: {np.degrees(np.arctan2(u - K[0,0]*0 - K[0,2], K[0,0])):.1f} deg (right +)")
    ok, rvec, tvec = cv2.solvePnP(marker.MARKER_CORNERS_3D, c.astype(np.float32), K,
                                  np.zeros(5), flags=cv2.SOLVEPNP_IPPE_SQUARE)
    t = tvec.ravel()
    print("tvec:", np.round(t, 3), f"| angle from TVEC: {np.degrees(np.arctan2(t[0], t[2])):.1f} deg")
    cv2.circle(img, (int(u), int(v)), 6, (255, 0, 0), 2)
    cv2.line(img, (int(K[0, 2]), 0), (int(K[0, 2]), img.shape[0]), (0, 255, 0), 1)
cv2.imwrite("pixels_check.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))