
import csv
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.sim import config
from src.sim.environment import Environment
from src.perception.marker import detect_marker

READINGS_PER_SPOT = 200
DISTANCES = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

print(f"Camera noise setting: {config.CAMERA_PIXEL_NOISE} pixels")
if config.CAMERA_PIXEL_NOISE == 0:
    print("Noise is OFF, so every reading will be identical. Set CAMERA_PIXEL_NOISE in config.py first.")

env = Environment(seed=0)
K = env.get_camera_intrinsics()
rng = np.random.default_rng(1)   # fixed seed: same numbers every run
dock_x, dock_y = env.get_dock_position()
marker_x, marker_y = dock_x - 0.005, dock_y

results = []
for d in DISTANCES:
    x, y = marker_x - d, -0.6                  # same lane as Task 2, clear of the orange box
    yaw = math.atan2(marker_y - y, marker_x - x)
    env.reset(x=x, y=y, yaw=yaw)
    env.step()
    image = env.get_camera_image()             # robot is parked, so the picture never changes

    ranges, bearings = [], []
    for _ in range(READINGS_PER_SPOT):
        reading = detect_marker(image, K, rng=rng)
        if reading is not None:
            ranges.append(reading[0])
            bearings.append(reading[1])

    mean_range = np.mean(ranges)
    results.append({
        "distance_setting": d,
        "readings": len(ranges),
        "mean_range": mean_range,
        "std_range": np.std(ranges),
        "std_bearing": np.std(bearings),
    })
    print(f"  {mean_range:.2f} m: range spread {np.std(ranges) * 100:.2f} cm,"
          f" bearing spread {np.std(bearings):.5f} rad  ({len(ranges)} readings)")

# ---- Fit: range spread = k * range^2 (least squares) ----
r = np.array([row["mean_range"] for row in results])
s = np.array([row["std_range"] for row in results])
k = np.sum(s * r ** 2) / np.sum(r ** 4)
bearing_std = np.mean([row["std_bearing"] for row in results])
print(f"\nRANGE_NOISE_K     = {k:.4f}    (range spread = k x range^2)")
print(f"BEARING_NOISE_STD = {bearing_std:.5f} rad (about the same at every distance)")
print("Copy these two numbers into the top of src/perception/marker.py")

with open("perception_noise.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

# ---- Graph ----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
smooth = np.linspace(0, r.max() * 1.05, 100)
ax1.plot(r, s * 100, "o", color="#1f6fb4", label="Measured")
ax1.plot(smooth, k * smooth ** 2 * 100, "-", color="#d0623a", label=f"Fit: {k:.4f} x range²")
ax1.set_xlabel("Range (m)")
ax1.set_ylabel("Range spread, 1 std (cm)")
ax1.set_title("Range noise grows with distance squared")
ax1.legend()
ax1.grid(alpha=0.3)
ax2.plot(r, [row["std_bearing"] for row in results], "o", color="#1f6fb4", label="Measured")
ax2.axhline(bearing_std, color="#d0623a", label=f"Average: {bearing_std:.4f} rad")
ax2.set_ylim(0, bearing_std * 2.5)
ax2.set_xlabel("Range (m)")
ax2.set_ylabel("Bearing spread, 1 std (rad)")
ax2.set_title("Bearing noise stays about the same")
ax2.legend()
ax2.grid(alpha=0.3)
fig.suptitle(f"Perception noise with {config.CAMERA_PIXEL_NOISE} px corner noise")
fig.tight_layout()
fig.savefig("perception_noise.png", dpi=150)
print("Saved perception_noise.csv and perception_noise.png")