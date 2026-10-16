import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np
from PIL import Image
from src.sim.environment import Environment
from src.perception.marker import detect_marker

env = Environment(seed=0)
K = env.get_camera_intrinsics()
dock = env.get_dock_position()
print("dock:", dock)

for x in (0.0, 0.8, 1.4, 1.8, 2.0):
    env.reset(x, 0.0, 0.0)           # facing +x, straight at the dock
    img = env.get_camera_image()
    Image.fromarray(img).save(f"detect_x{x:.1f}.png")
    print(f"x={x:.1f} (true range {dock[0]-x:.2f} m):", detect_marker(img, K))