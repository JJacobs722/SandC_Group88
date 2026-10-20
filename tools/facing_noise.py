"""How accurate is the marker's facing direction at different distances?

Parks the robot at 7 distances (0.5 to 3.5 m) and 9 spots each, facing the marker,
and compares detect_marker_pose with the true answer. Run once with pixel noise off
and once with it on. These numbers set FACING_NOISE_K in marker.py.

Ground truth is used here to grade the measurement only, never by the robot.
Run from the top folder:  python -m tools.facing_noise
"""
import math

import numpy as np

from src.sim import config
from src.sim.environment import Environment
from src.perception.marker import detect_marker_pose


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


env = Environment(seed=0)
K = env.get_camera_intrinsics()
for name in ("obstacle_1", "obstacle_2", "obstacle_3"):        # clear the view for this test
    env.m.geom_pos[env.m.geom(name).id][:2] = [-2.0, -2.0]
(cx, cy), psi = env.marker_frame_in_world()
rng = np.random.default_rng(0)

for noise in (0.0, 0.5):
    config.CAMERA_PIXEL_NOISE = noise
    print(f"\npixel noise {noise}")
    print("distance   facing error (deg): mean   spread   worst")
    for d in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5):
        errors = []
        for side in (-0.6, 0.0, 0.6):
            for look in (-0.3, 0.0, 0.3):
                # spot d metres out from the marker, side metres to one side, turned slightly
                wx = cx + math.cos(psi) * d - math.sin(psi) * side
                wy = cy + math.sin(psi) * d + math.cos(psi) * side
                env.reset(wx, wy, math.atan2(cy - wy, cx - wx) + look)
                env.step()
                image = env.get_camera_image()
                for _ in range(5 if noise else 1):
                    reading = detect_marker_pose(image, K, rng=rng)
                    if reading is None:
                        continue
                    true_facing = wrap(-env.get_ground_truth_marker_frame()[2])
                    errors.append(math.degrees(wrap(reading[2] - true_facing)))
        e = np.array(errors)
        print(f"  {d:.1f} m              {e.mean():+6.2f}   {e.std():6.2f}   {np.abs(e).max():6.1f}")
