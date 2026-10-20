"""Same pipeline as run_dock.py, shown as a 4 panel dashboard:
top-down view, robot camera, LiDAR scan and live numbers.

  python run_dashboard.py                      # show the window
  python run_dashboard.py --record             # also save results/dashboard.mp4
  python run_dashboard.py --headless --record  # no window, just save the video

Works with plain python on Windows and Mac (no mjpython needed).
Press q in the window to stop early.
"""
import sys
import time
import math
import importlib.util
from pathlib import Path

import cv2
import numpy as np
import mujoco

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))     # add src to path

# find and load the EKF file
_cands = [p for p in (ROOT / "src" / "estimation").glob("*.py")
          if "class EKF" in p.read_text(errors="ignore")]
if not _cands:
    raise FileNotFoundError("No file in src/estimation contains 'class EKF'.")
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

SHOW = "--headless" not in sys.argv
RECORD = "--record" in sys.argv
T_MAX = 180.0

# where WE put the robot in the room. The robot is NOT told this. Change it to try anywhere.
PLACE = (0.0, 0.0, 0.0)
DRAW_EVERY = 5            # redraw the dashboard every 5 ticks (10 times a second)
PW, PH = 480, 360         # size of each panel in pixels

# ---------------- setup (same as run_dock.py) ----------------
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
START_RANGE = 1.8
RELOCK_AFTER = 5
MARKER_XY = np.array([0.0, 0.0])
standoff_pt = np.array([0.60, 0.0])
stop_pt = np.array([0.25, 0.0])

nav = LidarNavigator()
Q = np.diag([20.0, 20.0, 5.0])
Rw = np.diag([1.0, 2.0])

cam_every = round(0.2 / dt)              # camera 5 Hz
lidar_every = round(0.1 / dt)            # LiDAR 10 Hz

# ---------------- dashboard drawing (display only) ----------------
overview = mujoco.Renderer(env.m, PH, PW)
top_cam = mujoco.MjvCamera()
top_cam.type = mujoco.mjtCamera.mjCAMERA_FREE
top_cam.lookat[:] = [0.0, 0.0, 0.0]
top_cam.distance = 7.2
top_cam.azimuth = 90
top_cam.elevation = -89.9

_aruco = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))
WHITE, GREY, GREEN, YELLOW, RED, CYAN = (255, 255, 255), (130, 130, 130), (80, 220, 80), (0, 220, 255), (60, 60, 230), (230, 200, 60)


def put(img, text, x, y, colour=WHITE, size=0.5, bold=1):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, size, colour, bold, cv2.LINE_AA)


def title(img, text):
    cv2.rectangle(img, (0, 0), (PW, 26), (40, 40, 40), -1)
    put(img, text, 8, 18, WHITE, 0.55, 1)


def add_sphere(scn, pos, radius, rgba):
    if scn.ngeom >= scn.maxgeom:
        return
    mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                        np.array([radius, 0, 0]), np.array(pos, float),
                        np.eye(3).flatten(), np.array(rgba, dtype=np.float32))
    scn.ngeom += 1


def draw_overview(x_hat, P, goal):
    overview.update_scene(env.d, camera=top_cam)
    scn = overview.scene
    if x_hat is None:                           # still searching: nothing to draw yet
        img = cv2.cvtColor(overview.render(), cv2.COLOR_RGB2BGR)
        title(img, "Top-down view (searching, no estimate yet)")
        return img
    # the EKF works in the marker frame, so turn it into room coordinates just to draw it
    xy = env.marker_to_world(x_hat[:2])
    gxy = env.marker_to_world(goal)
    psi = env.marker_frame_in_world()[1]
    c, s_ = math.cos(psi), math.sin(psi)
    turn = np.array([[c, -s_], [s_, c]])
    # green oval = 2 sigma position uncertainty from P
    vals, vecs = np.linalg.eigh(turn @ P[:2, :2] @ turn.T)
    axes = 2.0 * np.sqrt(np.maximum(vals, 1e-8))
    rot = np.eye(3)
    rot[:2, :2] = vecs
    if scn.ngeom < scn.maxgeom:
        mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_ELLIPSOID,
                            np.array([max(axes[0], 0.01), max(axes[1], 0.01), 0.004]),
                            np.array([xy[0], xy[1], 0.34]), rot.flatten(),
                            np.array([0.1, 0.9, 0.2, 0.45], dtype=np.float32))
        scn.ngeom += 1
    add_sphere(scn, [xy[0], xy[1], 0.35], 0.07, [0.1, 0.9, 0.2, 1.0])   # estimate
    add_sphere(scn, [gxy[0], gxy[1], 0.05], 0.04, [1.0, 0.85, 0.0, 1.0])    # current target
    img = cv2.cvtColor(overview.render(), cv2.COLOR_RGB2BGR)
    title(img, "Top-down view (green = EKF estimate, yellow = target)")
    return img


def draw_camera(image, det):
    if image is None:
        img = np.zeros((PH, PW, 3), np.uint8)
    else:
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        corners, ids, _ = _aruco.detectMarkers(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(bgr, corners, ids)
        img = cv2.resize(bgr, (PW, PH))
    title(img, "Robot camera (what perception sees)")
    if det is not None:
        put(img, f"marker 0 seen: {det[0]:.2f} m, {math.degrees(det[1]):+.1f} deg", 10, PH - 14, GREEN, 0.6, 2)
    else:
        put(img, "marker not visible", 10, PH - 14, YELLOW, 0.6, 2)
    return img


def draw_lidar(angles, ranges, info, det):
    img = np.zeros((PH, PW, 3), np.uint8)
    cx, cy, s = PW // 2, PH // 2 + 20, 48           # robot in the middle, 48 px per metre
    for r in (1, 2, 3):
        cv2.circle(img, (cx, cy), int(r * s), (50, 50, 50), 1)
        put(img, f"{r} m", cx + 4, cy - int(r * s) + 14, GREY, 0.4)

    def to_px(fwd, left):
        return int(cx - left * s), int(cy - fwd * s)

    if angles is not None:
        ok = np.isfinite(ranges)
        for a, r in zip(angles[ok], ranges[ok]):
            cv2.circle(img, to_px(r * math.cos(a), r * math.sin(a)), 2, RED, -1)
    cv2.circle(img, (cx, cy), int(nav.r_safe * s), GREY, 1)
    cv2.rectangle(img, to_px(0.14, 0.14), to_px(-0.14, -0.14), WHITE, 2)
    if info is not None:
        g = info["goal_phi"]
        cv2.line(img, (cx, cy), to_px(1.2 * math.cos(g), 1.2 * math.sin(g)), YELLOW, 1)
        p = info["phi"]
        cv2.arrowedLine(img, (cx, cy), to_px(1.0 * math.cos(p), 1.0 * math.sin(p)), GREEN, 2, tipLength=0.15)
    if det is not None:
        cv2.line(img, (cx, cy), to_px(det[0] * math.cos(det[1]), det[0] * math.sin(det[1])), CYAN, 1)
    title(img, "LiDAR scan (robot frame, forward = up)")
    put(img, "red = LiDAR hits   yellow = goal direction", 8, PH - 30, GREY, 0.42)
    put(img, "green arrow = chosen safe heading   cyan = marker", 8, PH - 12, GREY, 0.42)
    return img


def draw_numbers(x_hat, P, phase, det, info, v, w, wl, wr, stats):
    img = np.zeros((PH, PW, 3), np.uint8)
    title(img, "Live numbers (all from the robot's own sensors, no ground truth)")
    if x_hat is None:
        task = f"SEARCHING ({search.mode}): no idea where it is yet"
        est = [("EKF estimate", GREEN), ("  not started: marker not seen up close yet", WHITE), ("", WHITE), ("", WHITE)]
    else:
        task = f"DOCKING, {'1: final approach' if phase else '0: to stand-off point'}"
        sx, sy, sth = np.sqrt(np.diag(P))
        est = [("EKF estimate (marker frame: marker at 0,0)", GREEN),
               (f"  x, y          {x_hat[0]:+.2f} m, {x_hat[1]:+.2f} m", WHITE),
               (f"  heading       {math.degrees(x_hat[2]):+.1f} deg", WHITE),
               (f"  uncertainty   x {sx * 100:.1f} cm, y {sy * 100:.1f} cm, heading {math.degrees(sth):.1f} deg", WHITE)]
    lines = [
        (f"time            {env.time:6.1f} s", WHITE),
        (f"task            {task}", WHITE),
        ("", WHITE),
    ] + est + [
        ("", WHITE),
        ("Perception", CYAN),
        (f"  marker        {'seen' if det is not None else 'not visible'}", WHITE),
        (f"  readings      {stats['accepted']} accepted, {stats['rejected']} rejected by gate", WHITE),
        ("", WHITE),
        ("Control", YELLOW),
        (f"  distance to target {info['dist']:.2f} m" if info else "", WHITE),
        (f"  closest obstacle   {info['min_range']:.2f} m" if info else "", WHITE),
        (f"  speed {v:.3f} m/s, turn {w:+.2f} rad/s", WHITE),
        (f"  wheels  L {wl:+.2f}  R {wr:+.2f} rad/s", WHITE),
    ]
    y = 50
    for text, colour in lines:
        put(img, text, 12, y, colour, 0.5)
        y += 18
    return img


# ---------------- run ----------------
phase, goal = 0, standoff_pt             # 0 = go to stand-off point, 1 = final approach
angles = ranges = info = None
image = det = None
x_hat = P = None
k, log = 0, []
docked = False
found_at = None
rejects = 0
v = w = wl = wr = 0.0
stats = {"accepted": 0, "rejected": 0}
writer = None
if RECORD:
    (ROOT / "results").mkdir(exist_ok=True)
    writer = cv2.VideoWriter(str(ROOT / "results" / "dashboard.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), 1 / (DRAW_EVERY * dt), (2 * PW, 2 * PH))

while env.time < T_MAX:
    t0 = time.time()

    # ---- sensing ----
    if k % lidar_every == 0:
        angles, ranges = env.get_lidar()
    wheel_l, wheel_r = env.get_wheel_angles()

    # ---- perception ----
    if k % cam_every == 0:
        image = env.get_camera_image()
        det = detect_marker_pose(image, K_cam)
    else:
        det = None
    R_meas = marker_covariance(det[0]) if det is not None else None

    # ---- estimation ----
    odom_pose, _ = odom.step(wheel_l, wheel_r)                 # wheels only, start frame
    if ekf is None:
        if det is not None and det[0] < START_RANGE:
            ekf = start_from_marker(det, R_meas, cfg)          # first close sighting sets the pose
            ekf.step(wheel_l, wheel_r)
            x_hat, P = ekf.state, ekf.covariance
            found_at = env.time
        elif det is not None:
            search.saw_marker(odom_pose, det)                  # too far: head towards it first
    else:
        x_hat, P = ekf.step(wheel_l, wheel_r, det, None, R_meas)
        if det is not None:
            stats["accepted" if ekf.last_update_accepted else "rejected"] += 1
            rejects = 0 if ekf.last_update_accepted else rejects + 1
            if rejects >= RELOCK_AFTER and det[0] < START_RANGE:
                ekf = start_from_marker(det, R_meas, cfg)      # trust the camera again
                ekf.step(wheel_l, wheel_r)
                x_hat, P = ekf.state, ekf.covariance
                rejects = 0

    # ---- control ----
    if ekf is None:
        mode, target = search.update(odom_pose, angles, ranges, dt)
        if mode == "spin":
            v, w = 0.0, search.spin_rate
            info = None
        else:
            ref, v_cap, info = nav.plan(odom_pose, target, angles, ranges, search.marker_xy, v_max=0.15)
            v, w = control(odom_pose, ref, Q, Rw, dt)
            v = float(np.clip(v, 0.0, v_cap))
    else:
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

    # ---- evaluation only (ground truth never goes back into control, and is never shown live) ----
    if ekf is not None:
        gt = env.get_ground_truth_marker_frame()
        e = gt - x_hat
        e[2] = angle_wrap(e[2])
        log.append((env.time, np.hypot(*e[:2]), abs(e[2])))

    # ---- dashboard ----
    if k % DRAW_EVERY == 0:
        shown_det = det if det is not None else (detect_marker_pose(image, K_cam) if image is not None else None)
        frame = np.vstack([np.hstack([draw_overview(x_hat, P, goal), draw_camera(image, shown_det)]),
                           np.hstack([draw_lidar(angles, ranges, info, shown_det),
                                      draw_numbers(x_hat, P, phase, shown_det, info, v, w, wl, wr, stats)])])
        if writer is not None:
            writer.write(frame)
        if SHOW:
            cv2.imshow("SandC_Group88 dashboard", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            time.sleep(max(0.0, DRAW_EVERY * dt - (time.time() - t0)))
    k += 1

if writer is not None:
    writer.release()
    print("saved results/dashboard.mp4")

print(f"\ndocked = {docked}   time = {env.time:.1f} s   marker found at {'never' if found_at is None else f'{found_at:.1f} s'}")
print(f"camera readings: {stats['accepted']} accepted, {stats['rejected']} rejected by the gate")
if log:
    log = np.array(log)
    print("\n---- evaluation only (ground truth, never used by the robot) ----")
    print(f"final estimate error:   {log[-1, 1] * 100:.1f} cm, heading {math.degrees(log[-1, 2]):.1f} deg")
    print(f"average estimate error: {log[:, 1].mean() * 100:.1f} cm")
    print(f"worst estimate error:   {log[:, 1].max() * 100:.1f} cm")
    gt = env.get_ground_truth_marker_frame()
    print(f"true distance to stop point: {np.hypot(*(gt[:2] - stop_pt)) * 100:.1f} cm")

if SHOW:
    print("\npress any key in the window to close")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
