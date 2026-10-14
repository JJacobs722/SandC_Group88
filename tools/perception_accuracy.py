"""Task 2: how accurate is perception at different distances and angles?

Runs with camera noise OFF, so only the method's own errors show.
Compares corner refinement off vs on.
Saves perception_accuracy.csv and perception_accuracy.png
"""
import csv
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.sim import config
from src.sim.environment import Environment
import src.perception.marker as marker

config.CAMERA_PIXEL_NOISE = 0.0   # noise off for this test

env = Environment(seed=0)
K = env.get_camera_intrinsics()
dock_x, dock_y = env.get_dock_position()
marker_x, marker_y = dock_x - 0.005, dock_y


def measure(x, y, yaw):
    """Put the robot somewhere, detect, and return (my reading, true reading)."""
    env.reset(x=x, y=y, yaw=yaw)
    env.step()
    result = marker.detect_marker(env.get_camera_image(), K)
    # Ground truth: evaluation only
    tx, ty, tyaw = env.get_ground_truth()
    true_range = math.hypot(marker_x - tx, marker_y - ty)
    true_bearing = math.atan2(marker_y - ty, marker_x - tx) - tyaw
    return result, (true_range, true_bearing)


rows = []

# ---- Test A: distance. Robot drives back along a lane that avoids the orange box, always facing the marker ----
distances = [0.3, 0.5, 0.8, 1.0, 1.2, 1.6, 2.0, 2.4, 2.8, 3.2, 3.6, 4.0]
for refine in [False, True]:
    marker.set_corner_refinement(refine)
    for d in distances:
        x, y = marker_x - d, -0.6
        yaw = math.atan2(marker_y - y, marker_x - x)
        result, (tr, tb) = measure(x, y, yaw)
        rows.append({
            "test": "distance", "refine": refine, "setting": d,
            "true_range": tr, "true_bearing": tb,
            "range": None if result is None else result[0],
            "bearing": None if result is None else result[1],
        })

# ---- Test B: viewing angle. Robot 0.8 m away (clear of the orange box), moving round to the side, always facing the marker ----
marker.set_corner_refinement(True)
for angle_deg in [0, 15, 30, 45, 55, 65, 70, 75, 80]:
    a = math.radians(angle_deg)
    x, y = marker_x - 0.8 * math.cos(a), marker_y + 0.8 * math.sin(a)
    yaw = math.atan2(marker_y - y, marker_x - x)
    result, (tr, tb) = measure(x, y, yaw)
    rows.append({
        "test": "view_angle", "refine": True, "setting": angle_deg,
        "true_range": tr, "true_bearing": tb,
        "range": None if result is None else result[0],
        "bearing": None if result is None else result[1],
    })

# ---- Test C: marker towards the edge of the picture. Robot 0.9 m away, turning away ----
for turn_deg in [0, 10, 20, 25, 28, 30, 35]:
    x, y = 1.6, 0.4
    yaw = math.atan2(marker_y - y, marker_x - x) - math.radians(turn_deg)
    result, (tr, tb) = measure(x, y, yaw)
    rows.append({
        "test": "edge_of_view", "refine": True, "setting": turn_deg,
        "true_range": tr, "true_bearing": tb,
        "range": None if result is None else result[0],
        "bearing": None if result is None else result[1],
    })

# ---- Save every reading ----
with open("perception_accuracy.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

# ---- Print a summary ----
for test, label in [("distance", "Distance test (true distance shown)"),
                    ("view_angle", "Viewing angle test (degrees off square-on)"),
                    ("edge_of_view", "Edge of view test (degrees turned away)")]:
    print(f"\n{label}")
    for r in rows:
        if r["test"] != test:
            continue
        tag = "refine ON " if r["refine"] else "refine OFF"
        where = f"{r['true_range']:.2f} m" if test == "distance" else f"{r['setting']} deg"
        if r["range"] is None:
            print(f"  {tag}  {where:>7}   not seen")
        else:
            print(f"  {tag}  {where:>7}   range error {(r['range'] - r['true_range']) * 100:+6.1f} cm"
                  f"   bearing error {r['bearing'] - r['true_bearing']:+.4f} rad")

# ---- Graph: range error against distance, refinement off vs on ----
fig, ax = plt.subplots(figsize=(7, 4))
for refine, colour, label in [(False, "#999999", "Corner refinement off"),
                              (True, "#1f6fb4", "Corner refinement on")]:
    pts = [(r["true_range"], (r["range"] - r["true_range"]) * 100)
           for r in rows if r["test"] == "distance" and r["refine"] == refine and r["range"] is not None]
    ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", color=colour, label=label)
ax.axhline(0, color="black", linewidth=0.5)
ax.set_xlabel("Distance to marker (m)")
ax.set_ylabel("Range error (cm)")
ax.set_title("Perception range error, camera noise off")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("perception_accuracy.png", dpi=150)
print("\nSaved perception_accuracy.csv and perception_accuracy.png")