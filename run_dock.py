import sys
import time
import importlib.util
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))     # add src to path

# find and load the EKF file
_cands = [p for p in (ROOT / "src" / "estimation").glob("*.py")
          if "class EKF" in p.read_text(errors="ignore")]
if not _cands:
    raise FileNotFoundError("No file in src/estimation contains 'class EKF'. "
                            "Copy the real one from origin/feature/estimation.")
_spec = importlib.util.spec_from_file_location("ekf_module", _cands[0])
ekf_module = importlib.util.module_from_spec(_spec)
sys.modules["ekf_module"] = ekf_module
_spec.loader.exec_module(ekf_module)
EKF, EKF_build, angle_wrap = ekf_module.EKF, ekf_module.EKF_build, ekf_module.angle_wrap

from src.sim.environment import Environment
from src.sim import config
from src.perception.marker import detect_marker, measurement_covariance
from src.control.lqr_tracker import control, to_wheels
from src.control.lidar_nav import LidarNavigator

REALTIME = True          # False = run as fast as possible
T_MAX = 150.0

# ---------------- setup ----------------
env = Environment(seed=0)
START = (0.0, 0.0, 0.0)                  # assumed start pose
env.reset(*START)

dock = env.get_dock_position()           # dock position (known map)
K_cam = env.get_camera_intrinsics()
dt = config.CONTROL_DT

cfg = EKF_build(wheel_radius=0.033, wheel_base=0.288)
ekf = EKF(np.array(START) + [0.03, -0.03, 0.05], cfg)   # start the EKF slightly off on purpose
ekf.step(*env.get_wheel_angles())

STANDOFF, STOP_DIST = 0.60, 0.25
u = np.array(START[:2]) - dock
u /= np.linalg.norm(u)
standoff_pt = dock + STANDOFF * u
stop_pt = dock + STOP_DIST * u

nav = LidarNavigator()
Q = np.diag([20.0, 20.0, 5.0])
Rw = np.diag([1.0, 2.0])

cam_every = round(0.2 / dt)              # camera 5 Hz
lidar_every = round(0.1 / dt)            # LiDAR 10 Hz


def draw_estimate(viewer, x_hat, P):
    # green dot = EKF estimate, green ellipse = uncertainty (display only)
    scn = viewer.user_scn
    vals, vecs = np.linalg.eigh(P[:2, :2])
    axes = 2.0 * np.sqrt(np.maximum(vals, 1e-8))          # 2 sigma
    rot = np.eye(3)
    rot[:2, :2] = vecs
    mujoco.mjv_initGeom(scn.geoms[0], mujoco.mjtGeom.mjGEOM_SPHERE,
                        np.array([0.03, 0.0, 0.0]), np.array([x_hat[0], x_hat[1], 0.40]),
                        np.eye(3).flatten(), np.array([0.1, 0.9, 0.2, 1.0], dtype=np.float32))
    mujoco.mjv_initGeom(scn.geoms[1], mujoco.mjtGeom.mjGEOM_ELLIPSOID,
                        np.array([max(axes[0], 0.01), max(axes[1], 0.01), 0.004]),
                        np.array([x_hat[0], x_hat[1], 0.02]),
                        rot.flatten(), np.array([0.1, 0.9, 0.2, 0.35], dtype=np.float32))
    scn.ngeom = 2


phase, goal = 0, standoff_pt             # 0 = go to stand-off point, 1 = final approach
angles = ranges = None
misses, k, log = 0, 0, []
docked = False

with mujoco.viewer.launch_passive(env.m, env.d) as viewer:
    viewer.cam.lookat[:] = [0.0, 0.0, 0.0]       # top down view of room
    viewer.cam.distance = 6.5
    viewer.cam.azimuth = 90
    viewer.cam.elevation = -70

    while viewer.is_running() and env.time < T_MAX:
        t0 = time.time()

        # ---- sensing ----
        if k % lidar_every == 0:
            angles, ranges = env.get_lidar()

        # ---- perception ----
        det = detect_marker(env.get_camera_image(), K_cam) if k % cam_every == 0 else None
        R_meas = measurement_covariance(det[0]) if det is not None else None

        # ---- estimation ----
        x_hat, P = ekf.step(*env.get_wheel_angles(), det, dock, R_meas)
        if det is not None:
            misses = 0 if ekf.last_update_accepted else misses + 1

        # ---- control (only uses the estimate) ----
        if phase == 0 and np.hypot(*(goal - x_hat[:2])) < 0.10:
            phase, goal = 1, stop_pt
        if phase == 1 and np.hypot(*(stop_pt - x_hat[:2])) < 0.04:
            docked = True
            env.set_wheel_speeds(0.0, 0.0)
            break

        ref, v_cap, info = nav.plan(x_hat, goal, angles, ranges, dock,
                                    v_max=0.15 if phase == 0 else 0.08)
        v, w = control(x_hat, ref, Q, Rw, dt)            # LQR
        v = float(np.clip(v, 0.0, v_cap))                # lidar speed cap, no reversing
        wl, wr = to_wheels(v, w)
        env.set_wheel_speeds(wl, wr)
        env.step()

        # ---- evaluation only (ground truth never goes back into control) ----
        gt = env.get_ground_truth()
        e = gt - x_hat
        e[2] = angle_wrap(e[2])
        log.append((env.time, np.hypot(*e[:2]), float(e @ np.linalg.solve(P, e)), info["min_range"]))

        with viewer.lock():
            draw_estimate(viewer, x_hat, P)
        viewer.sync()

        if k % 25 == 0:
            print(f"t={env.time:5.1f} ph={phase} det={'Y' if det is not None else '-'} "
                  f"acc={ekf.last_update_accepted} "
                  f"est=({x_hat[0]:.2f},{x_hat[1]:.2f}) TRUE=({gt[0]:.2f},{gt[1]:.2f}) "
                  f"lidar_min={info['min_range']:.2f} v={v:.3f} w={w:.3f}")
        k += 1

        if REALTIME:
            time.sleep(max(0.0, dt - (time.time() - t0)))

    print(f"\ndocked={docked}")
    if log:
        log = np.array(log)
        print(f"t={log[-1,0]:.1f}s  est err {log[-1,1]:.3f} m  mean NEES {log[:,2].mean():.2f} "
              f"min lidar range {log[:,3].min():.2f} m")

    # keep window open at the end
    while viewer.is_running():
        viewer.sync()
        time.sleep(0.05)