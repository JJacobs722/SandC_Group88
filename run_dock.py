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
start_from_marker = ekf_module.start_from_marker

from src.sim.environment import Environment
from src.sim import config
from src.perception.marker import detect_marker_pose, marker_covariance
from src.control.lqr_tracker import control, to_wheels
from src.control.lidar_nav import LidarNavigator
from src.control.search import MarkerSearch

REALTIME = True          # False = run as fast as possible
T_MAX = 180.0

# where WE put the robot in the room. The robot is NOT told this. Change it to try anywhere.
PLACE = (0.0, 0.0, 0.0)

# ---------------- setup ----------------
env = Environment(seed=0)
env.reset(*PLACE)

K_cam = env.get_camera_intrinsics()
dt = config.CONTROL_DT
# calibrated wheel numbers (tools/calibrate_wheels.py), k_wheel tuned in evaluate.py
cfg = EKF_build(wheel_radius=0.0281, wheel_base=0.351, k_wheel=5e-3)

# before the marker is seen the robot only has its wheels, in its own start frame
odom = EKF([0.0, 0.0, 0.0], cfg)          # start frame: "wherever I started is (0, 0)"
odom.step(*env.get_wheel_angles())
search = MarkerSearch()

# once the marker is seen up close, the EKF starts in the MARKER frame:
# marker at (0, 0), its face pointing along +x. No map needed, the marker defines it.
ekf = None
START_RANGE = 1.8                        # m: only start the EKF this close (marker facing is noisy further away)
RELOCK_AFTER = 5                         # camera rejected this many times in a row -> restart the EKF from the camera
MARKER_XY = np.array([0.0, 0.0])
standoff_pt = np.array([0.60, 0.0])      # 0.60 m straight out from the marker
stop_pt = np.array([0.25, 0.0])          # 0.25 m straight out from the marker

nav = LidarNavigator()
Q = np.diag([20.0, 20.0, 5.0])
Rw = np.diag([1.0, 2.0])

cam_every = round(0.2 / dt)              # camera 5 Hz
lidar_every = round(0.1 / dt)            # LiDAR 10 Hz


def draw_estimate(viewer, x_hat, P):
    # green dot = EKF estimate, green ellipse = uncertainty (display only)
    # the EKF works in the marker frame, so turn it into room coordinates just to draw it
    scn = viewer.user_scn
    xy = env.marker_to_world(x_hat[:2])
    psi = env.marker_frame_in_world()[1]
    c, s = np.cos(psi), np.sin(psi)
    turn = np.array([[c, -s], [s, c]])
    vals, vecs = np.linalg.eigh(turn @ P[:2, :2] @ turn.T)
    axes = 2.0 * np.sqrt(np.maximum(vals, 1e-8))          # 2 sigma
    rot = np.eye(3)
    rot[:2, :2] = vecs
    mujoco.mjv_initGeom(scn.geoms[0], mujoco.mjtGeom.mjGEOM_SPHERE,
                        np.array([0.03, 0.0, 0.0]), np.array([xy[0], xy[1], 0.40]),
                        np.eye(3).flatten(), np.array([0.1, 0.9, 0.2, 1.0], dtype=np.float32))
    mujoco.mjv_initGeom(scn.geoms[1], mujoco.mjtGeom.mjGEOM_ELLIPSOID,
                        np.array([max(axes[0], 0.01), max(axes[1], 0.01), 0.004]),
                        np.array([xy[0], xy[1], 0.02]),
                        rot.flatten(), np.array([0.1, 0.9, 0.2, 0.35], dtype=np.float32))
    scn.ngeom = 2


phase, goal = 0, standoff_pt             # 0 = go to stand-off point, 1 = final approach
angles = ranges = None
x_hat = P = None
k, log = 0, []
rejects = 0
docked = False
found_at = None

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
        wheel_l, wheel_r = env.get_wheel_angles()

        # ---- perception ----
        det = detect_marker_pose(env.get_camera_image(), K_cam) if k % cam_every == 0 else None
        R_meas = marker_covariance(det[0]) if det is not None else None

        # ---- estimation ----
        odom_pose, _ = odom.step(wheel_l, wheel_r)                 # wheels only, start frame
        if ekf is None:
            if det is not None and det[0] < START_RANGE:
                ekf = start_from_marker(det, R_meas, cfg)          # first close sighting sets the pose
                ekf.step(wheel_l, wheel_r)                         # just records the wheel angles
                x_hat, P = ekf.state, ekf.covariance
                found_at = env.time
                print(f"t={env.time:5.1f} marker found {det[0]:.2f} m away, EKF started at "
                      f"({x_hat[0]:.2f}, {x_hat[1]:.2f}, {np.degrees(x_hat[2]):.0f} deg) in the marker frame")
            elif det is not None:
                search.saw_marker(odom_pose, det)                  # too far: head towards it first
        else:
            x_hat, P = ekf.step(wheel_l, wheel_r, det, None, R_meas)
            if det is not None:
                rejects = 0 if ekf.last_update_accepted else rejects + 1
                if rejects >= RELOCK_AFTER and det[0] < START_RANGE:
                    # the wheels have drifted so far the gate keeps saying no: trust the camera again
                    ekf = start_from_marker(det, R_meas, cfg)
                    ekf.step(wheel_l, wheel_r)
                    x_hat, P = ekf.state, ekf.covariance
                    rejects = 0
                    print(f"t={env.time:5.1f} camera rejected {RELOCK_AFTER} times in a row, EKF restarted from the marker")

        # ---- control ----
        if ekf is None:
            # searching: no idea where the marker is yet
            mode, target = search.update(odom_pose, angles, ranges, dt)
            if mode == "spin":
                v, w = 0.0, search.spin_rate
            else:
                ref, v_cap, info = nav.plan(odom_pose, target, angles, ranges, search.marker_xy, v_max=0.15)
                v, w = control(odom_pose, ref, Q, Rw, dt)
                v = float(np.clip(v, 0.0, v_cap))
        else:
            # docking: only uses the estimate, in the marker frame
            if phase == 0 and np.hypot(*(goal - x_hat[:2])) < 0.10:
                phase, goal = 1, stop_pt
            if phase == 1 and np.hypot(*(stop_pt - x_hat[:2])) < 0.04:
                docked = True
                env.set_wheel_speeds(0.0, 0.0)
                break
            ref, v_cap, info = nav.plan(x_hat, goal, angles, ranges, MARKER_XY,
                                        v_max=0.15 if phase == 0 else 0.08)
            v, w = control(x_hat, ref, Q, Rw, dt)            # LQR
            v = float(np.clip(v, 0.0, v_cap))                # lidar speed cap, no reversing
        wl, wr = to_wheels(v, w)
        env.set_wheel_speeds(wl, wr)
        env.step()

        # ---- evaluation only (ground truth never goes back into control) ----
        if ekf is not None:
            gt = env.get_ground_truth_marker_frame()
            e = gt - x_hat
            e[2] = angle_wrap(e[2])
            log.append((env.time, np.hypot(*e[:2]), float(e @ np.linalg.solve(P, e))))

        with viewer.lock():
            if ekf is not None:
                draw_estimate(viewer, x_hat, P)
        viewer.sync()

        if k % 25 == 0:
            if ekf is None:
                print(f"t={env.time:5.1f} searching ({search.mode}) det={'Y' if det is not None else '-'} "
                      f"v={v:.3f} w={w:.3f}")
            else:
                print(f"t={env.time:5.1f} ph={phase} det={'Y' if det is not None else '-'} "
                      f"acc={ekf.last_update_accepted} est=({x_hat[0]:.2f},{x_hat[1]:.2f}) "
                      f"v={v:.3f} w={w:.3f}")
        k += 1

        if REALTIME:
            time.sleep(max(0.0, dt - (time.time() - t0)))

    print(f"\ndocked={docked}  time={env.time:.1f}s  marker found at "
          f"{'never' if found_at is None else f'{found_at:.1f}s'}")
    if log:
        log = np.array(log)
        gt = env.get_ground_truth_marker_frame()
        print("---- evaluation only ----")
        print(f"est err {log[-1,1]:.3f} m  mean NEES {log[:,2].mean():.2f}  "
              f"true distance to stop point {np.hypot(*(gt[:2] - stop_pt)):.3f} m")

    # keep window open at the end
    while viewer.is_running():
        viewer.sync()
        time.sleep(0.05)
