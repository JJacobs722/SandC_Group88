"""FIRST thing to run at the lab: is the real robot talking to us?

  python -m tools.hw_check          sensors only, the robot does NOT move
  python -m tools.hw_check --move   also drives about 10 cm forward then turns about 45 deg, slowly

Saves hw_camera.png (what the camera sees) and hw_lidar.png (the scan from above).
Keep a hand near the robot's power switch for the --move test.
"""
import math
import sys
import time

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.hw.real_robot import RealRobot
from src.control.lqr_tracker import to_wheels, R_WHEEL, TRACK

env = RealRobot()
try:
    # ---- wheels ----
    l0, r0 = env.get_wheel_angles()
    print(f"wheel angles: left {l0:.3f} rad, right {r0:.3f} rad")

    # ---- LiDAR ----
    angles, ranges = env.get_lidar()
    ok = np.isfinite(ranges)
    print(f"LiDAR: {ok.sum()} of 360 directions see something, closest {ranges[ok].min():.2f} m "
          f"at {math.degrees(angles[ok][np.argmin(ranges[ok])]):+.0f} deg (0 = straight ahead, + = left)")
    ax = plt.subplot(projection="polar")
    ax.plot(angles[ok], ranges[ok], ".", markersize=2)
    ax.set_theta_zero_location("N")                      # forward = up
    ax.set_title("Real LiDAR (robot frame, forward = up)")
    plt.savefig("hw_lidar.png")

    # ---- camera ----
    img = env.get_camera_image()
    cv2.imwrite("hw_camera.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(f"camera: {img.shape[1]} x {img.shape[0]} from {env.camera_topic}, saved hw_camera.png")
    try:
        from src.perception.marker import detect_marker_pose, set_camera_setup
        from src.hw import hw_config as hw
        set_camera_setup(*env.get_camera_mount(), env.get_camera_distortion(), marker_size=hw.MARKER_SIZE)
        det = detect_marker_pose(img, env.get_camera_intrinsics())
        if det is None:
            print("marker: not visible (point the robot at it to test perception)")
        else:
            print(f"marker: {det[0]:.3f} m away, {math.degrees(det[1]):+.1f} deg, facing {math.degrees(det[2]):+.1f} deg"
                  "  <- check the distance with a tape measure")
    except RuntimeError as err:
        print("marker: skipped,", err)

    # ---- small, slow move ----
    if "--move" in sys.argv:
        input("\nClear 30 cm in front of the robot, then press Enter to drive forward about 10 cm...")
        l0, r0 = env.get_wheel_angles()
        for _ in range(100):                             # 2 s at 0.05 m/s (times the safety scale)
            env.set_wheel_speeds(*to_wheels(0.05, 0.0))
            env.step()
        env.stop()
        l1, r1 = env.get_wheel_angles()
        moved = R_WHEEL * ((l1 - l0) + (r1 - r0)) / 2
        print(f"wheels say it drove {moved * 100:.1f} cm. Check with a ruler: they should roughly agree.")

        input("Press Enter to turn left about 45 deg on the spot...")
        l0, r0 = env.get_wheel_angles()
        for _ in range(100):                             # 2 s at 0.4 rad/s (times the safety scale)
            env.set_wheel_speeds(*to_wheels(0.0, 0.4))
            env.step()
        env.stop()
        l1, r1 = env.get_wheel_angles()
        turned = R_WHEEL * ((r1 - r0) - (l1 - l0)) / TRACK
        print(f"wheels say it turned {math.degrees(turned):.0f} deg LEFT. If it turned right, the wheel names are swapped.")
finally:
    env.close()
    print("robot stopped")
