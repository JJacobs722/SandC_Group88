"""Task 1 check: does detect_marker give the right range and bearing?"""
import math

import cv2

from src.sim.environment import Environment
from src.perception.marker import detect_marker

env = Environment(seed=0)
K = env.get_camera_intrinsics()

# The marker's painted face is 5 mm in front of the dock's centre (the plate is 1 cm thick)
dock_x, dock_y = env.get_dock_position()
marker_x, marker_y = dock_x - 0.005, dock_y

# A few places to put the robot: (x, y, yaw)
test_poses = [
    (1.6, 0.0, 0.0),     # straight in front of the marker
    (1.6, 0.4, -0.4),    # off to the left, turned to face it
    (1.0, -0.8, 0.6),    # further away, off to the right
    (0.0, 0.0, 0.0),     # start position: the orange box should hide the marker
]

for x, y, yaw in test_poses:
    env.reset(x=x, y=y, yaw=yaw)
    env.step()
    result = detect_marker(env.get_camera_image(), K)

    # Ground truth: ONLY used here to check the answer, never in the real pipeline
    true_x, true_y, true_yaw = env.get_ground_truth()
    true_range = math.hypot(marker_x - true_x, marker_y - true_y)
    true_bearing = math.atan2(marker_y - true_y, marker_x - true_x) - true_yaw

    print(f"Robot at ({x}, {y}, yaw {yaw})")
    if result is None:
        print("   Marker not seen")
    else:
        rng_m, brg = result
        print(f"   Mine:  range {rng_m:.3f} m   bearing {brg:+.4f} rad")
        print(f"   True:  range {true_range:.3f} m   bearing {true_bearing:+.4f} rad")
        print(f"   Error: range {(rng_m - true_range) * 100:+.1f} cm   bearing {brg - true_bearing:+.4f} rad")

# Save what the camera saw from the second pose, so you can look at it
env.reset(x=1.6, y=0.4, yaw=-0.4)
env.step()
cv2.imwrite("perception_test.png", cv2.cvtColor(env.get_camera_image(), cv2.COLOR_RGB2BGR))
print("Saved perception_test.png")