import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np
import mujoco
from src.sim.environment import Environment

env = Environment(seed=0)
env.reset(0.0, 0.0, 0.0)
m, d = env.m, env.d

print("--- all bodies (name, world pos) ---")
for b in range(m.nbody):
    print(f"{m.body(b).name or '(world)':24s} {np.round(d.xpos[b], 2)}")

print("\n--- geoms attached to the world (walls, obstacles, floor) ---")
for g in range(m.ngeom):
    if m.geom_bodyid[g] == 0:
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "-"
        print(f"{name:20s} type={m.geom_type[g]} pos={np.round(d.geom_xpos[g], 2)} "
              f"size={np.round(m.geom_size[g], 2)}")

angles, ranges = env.get_lidar()
print("\n--- lidar, forward sector ---")
for deg in (-30, -15, 0, 15, 30):
    i = int(np.argmin(np.abs(angles - np.radians(deg))))
    print(f"{deg:4d} deg: {ranges[i]:.2f} m")
finite = ranges[np.isfinite(ranges)]
print("nearest return:", f"{finite.min():.2f} m" if finite.size else "none")