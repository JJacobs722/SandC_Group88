import sys
import csv
import time
import importlib.util
from pathlib import Path

import numpy as np
import mujoco

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

_cands = [p for p in (ROOT / "src" / "estimation").glob("*.py")
          if "class EKF" in p.read_text(errors="ignore")]
if not _cands:
    raise FileNotFoundError("No file in src/estimation contains 'class EKF'.")
_spec = importlib.util.spec_from_file_location("ekf_module", _cands[0])
ekf_module = importlib.util.module_from_spec(_spec)
sys.modules["ekf_module"] = ekf_module
_spec.loader.exec_module(ekf_module)
EKF, EKF_build, angle_wrap = ekf_module.EKF, ekf_module.EKF_build, ekf_module.angle_wrap

from src.sim.environment import Environment
from src.sim import config                      # used by marker.py
import sim.config as env_config
from src.perception.marker import detect_marker, measurement_covariance
from src.control.lqr_tracker import control, to_wheels
from src.control.lidar_nav import LidarNavigator

# ---------------- experiment settings ----------------
N_TRIALS = 20           # per condition; use 3 for a quick check
T_MAX = 120.0           # s; a trial not docked by then is a timeout
SUCCESS_DIST = 0.10     # m, true distance that counts as docked
STANDOFF, STOP_DIST = 0.60, 0.25
Q = np.diag([20.0, 20.0, 5.0])
Rw = np.diag([1.0, 2.0])

DEFAULTS = dict(pixel_noise=0.5, k_wheel=5e-3, enc_noise=0.002, lidar_noise=0.01,
                init_scale=1.0, p0_scale=1.0)
CONDITIONS = [
    # --- k_wheel sweep (estimator consistency vs responsiveness) ---
    dict(name="kw1e-3_nopx",   pixel_noise=0.0, k_wheel=1e-3),
    dict(name="kw1e-3",        k_wheel=1e-3),
    dict(name="kw5e-3",        k_wheel=5e-3),
    dict(name="kw2e-2",        k_wheel=2e-2),
    # --- stress: sensing ---
    dict(name="px2.0",         pixel_noise=2.0),
    dict(name="lidar_x5",      lidar_noise=0.05),
    # --- stress: odometry ---
    dict(name="enc_x10",       enc_noise=0.02),
    # --- bad initial guess ---
    dict(name="init_x6",       init_scale=6.0),
    dict(name="init_x6_P0x6",  init_scale=6.0, p0_scale=6.0),
]
for c in CONDITIONS:
    for key, val in DEFAULTS.items():
        c.setdefault(key, val)

dt = config.CONTROL_DT
cam_every, lidar_every = round(0.2 / dt), round(0.1 / dt)


# ---------------- evaluation only helpers ----------------
def sample_layout(env, rng, start_xy, dock):
    # random obstacle positions
    m = env.m
    ids = [m.geom(n).id for n in ("obstacle_1", "obstacle_2", "obstacle_3")]
    rad = [np.hypot(*m.geom_size[g][:2]) if m.geom_type[g] == 6 else m.geom_size[g][0] for g in ids]
    for _ in range(2000):
        pts = [np.array([rng.uniform(0.9, 1.5), rng.uniform(-0.3, 0.3)]),   # always near the direct line
               np.array([rng.uniform(0.5, 2.0), rng.uniform(-1.8, 1.8)]),
               np.array([rng.uniform(0.5, 2.0), rng.uniform(-1.8, 1.8)])]
        ok = all(np.linalg.norm(p - start_xy) > r + 0.5 and np.linalg.norm(p - dock) > r + 0.7
                 for p, r in zip(pts, rad))
        ok = ok and all(np.linalg.norm(pts[i] - pts[j]) > rad[i] + rad[j] + 0.55
                        for i in range(3) for j in range(i + 1, 3))
        if ok:
            return dict(zip(ids, pts))
    raise RuntimeError("could not place obstacles; loosen the constraints")


def collided(env):
    """True if the robot touches an obstacle or wall (MuJoCo contacts)."""
    m, d = env.m, env.d
    for i in range(d.ncon):
        c = d.contact[i]
        b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
        if (b1 == 0) != (b2 == 0):
            wg = c.geom1 if b1 == 0 else c.geom2
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, wg) or ""
            if name.startswith(("obstacle", "wall")):
                return True
    return False


# ---------------- one trial ----------------
def run_trial(env, cond, trial):
    rng = np.random.default_rng(trial)                        # same layout/start for every condition
    start = np.array([rng.uniform(0.0, 0.3), rng.uniform(-0.3, 0.3), rng.uniform(-0.35, 0.35)])
    dock = env.get_dock_position()
    for g, p in sample_layout(env, rng, start[:2], dock).items():
        env.m.geom_pos[g][:2] = p

    config.CAMERA_PIXEL_NOISE = cond["pixel_noise"]
    env_config.ENCODER_NOISE = cond["enc_noise"]
    env_config.LIDAR_NOISE = cond["lidar_noise"]
    env.rng = np.random.default_rng(10_000 + trial)           # encoder / LiDAR noise stream
    cam_rng = np.random.default_rng(20_000 + trial)           # pixel noise stream
    env.reset(*start)
    K_cam = env.get_camera_intrinsics()

    cfg = EKF_build(wheel_radius=0.033, wheel_base=0.288, k_wheel=cond["k_wheel"])
    init_err = np.array([0.03, 0.03, 0.04]) * cond["init_scale"]
    ekf = EKF(start + rng.normal(0, init_err), cfg, P0=cfg.P0 * cond["p0_scale"] ** 2)
    ekf.step(*env.get_wheel_angles())

    u = start[:2] - dock
    u /= np.linalg.norm(u)
    standoff_pt, stop_pt = dock + STANDOFF * u, dock + STOP_DIST * u
    nav = LidarNavigator()

    phase, goal = 0, standoff_pt
    angles = ranges = None
    k, docked, hit = 0, False, False
    nees, n_cam, n_rej, min_lidar, first_fix = [], 0, 0, np.inf, np.nan

    while env.time < T_MAX:
        if k % lidar_every == 0:
            angles, ranges = env.get_lidar()
        det = detect_marker(env.get_camera_image(), K_cam, rng=cam_rng) if k % cam_every == 0 else None
        R_meas = measurement_covariance(det[0]) if det is not None else None
        x_hat, P = ekf.step(*env.get_wheel_angles(), det, dock, R_meas)
        if det is not None:
            n_cam += 1
            if ekf.last_update_accepted:
                if np.isnan(first_fix):
                    first_fix = env.time
            else:
                n_rej += 1

        if phase == 0 and np.hypot(*(goal - x_hat[:2])) < 0.10:
            phase, goal = 1, stop_pt
        if phase == 1 and np.hypot(*(stop_pt - x_hat[:2])) < 0.04:
            docked = True
            break

        ref, v_cap, info = nav.plan(x_hat, goal, angles, ranges, dock,
                                    v_max=0.15 if phase == 0 else 0.08)
        v, w = control(x_hat, ref, Q, Rw, dt)                 # LQR on the estimate
        v = float(np.clip(v, 0.0, v_cap))
        wl, wr = to_wheels(v, w)
        env.set_wheel_speeds(wl, wr)
        env.step()

        # ---- evaluation only, not fed back into control ----
        gt = env.get_ground_truth()
        e = gt - x_hat
        e[2] = angle_wrap(e[2])
        nees.append(float(e @ np.linalg.solve(P, e)))
        min_lidar = min(min_lidar, info["min_range"])
        hit = hit or collided(env)
        k += 1

    env.set_wheel_speeds(0.0, 0.0)
    gt = env.get_ground_truth()
    e = gt - x_hat
    e[2] = angle_wrap(e[2])
    to_dock = np.arctan2(dock[1] - stop_pt[1], dock[0] - stop_pt[0])
    true_dist = float(np.linalg.norm(gt[:2] - stop_pt))
    success = bool(docked and true_dist < SUCCESS_DIST and not hit)

    if success:
        cause = "ok"
    elif hit:
        cause = "collision"
    elif not docked:
        cause = "timeout"
    else:
        cause = "false_dock"          # estimate says docked, true position says otherwise

    return dict(condition=cond["name"], trial=trial, success=success, failure_cause=cause,
                collision=hit, docked_by_estimate=docked, time_s=round(env.time, 2),
                true_dist_to_stop_m=round(true_dist, 4),
                heading_err_deg=round(abs(np.degrees(angle_wrap(gt[2] - to_dock))), 1),
                est_err_m=round(float(np.hypot(*e[:2])), 4),
                mean_nees=round(float(np.mean(nees)) if nees else np.nan, 2),
                cam_frames=n_cam, gate_rejects=n_rej, first_fix_s=round(float(first_fix), 2),
                min_lidar_m=round(float(min_lidar), 3))


# ---------------- run everything ----------------
def ms(x):
    x = np.array(x, float)
    x = x[~np.isnan(x)]
    return f"{x.mean():.3f}±{x.std():.3f}" if x.size else "n/a"


if __name__ == "__main__":
    env = Environment(seed=0)
    rows = []
    for cond in CONDITIONS:
        for trial in range(N_TRIALS):
            t0 = time.time()
            row = run_trial(env, cond, trial)
            rows.append(row)
            print(f"{cond['name']:14s} trial {trial:2d}  {row['failure_cause']:10s} "
                  f"dist={row['true_dist_to_stop_m']:.3f}  NEES={row['mean_nees']}  "
                  f"rej={row['gate_rejects']}/{row['cam_frames']}  ({time.time() - t0:.0f}s)")

    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    with open(out / "eval_trials.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    summary = []
    print("\n=== summary (mean ± std over trials) ===")
    for cond in CONDITIONS:
        r = [x for x in rows if x["condition"] == cond["name"]]
        n_ok = sum(x["success"] for x in r)
        causes = {c: sum(x["failure_cause"] == c for x in r) for c in ("collision", "timeout", "false_dock")}
        nees = [x["mean_nees"] for x in r]
        rej, cam = sum(x["gate_rejects"] for x in r), sum(x["cam_frames"] for x in r)
        print(f"{cond['name']:14s} success {n_ok}/{len(r)} | fails {causes} | "
              f"true dist {ms([x['true_dist_to_stop_m'] for x in r])} m | "
              f"heading err {ms([x['heading_err_deg'] for x in r])} deg | "
              f"est err {ms([x['est_err_m'] for x in r])} m | "
              f"NEES mean {np.nanmean(nees):.2f} median {np.nanmedian(nees):.2f} (want ~3) | "
              f"gate rejects {rej}/{cam} | first fix {ms([x['first_fix_s'] for x in r])} s | "
              f"min lidar {min(x['min_lidar_m'] for x in r):.2f} m")
        summary.append(dict(condition=cond["name"], trials=len(r), successes=n_ok, **causes,
                            nees_mean=round(float(np.nanmean(nees)), 2),
                            nees_median=round(float(np.nanmedian(nees)), 2),
                            gate_rejects=rej, cam_frames=cam,
                            true_dist_mean=round(float(np.mean([x["true_dist_to_stop_m"] for x in r])), 4),
                            est_err_mean=round(float(np.mean([x["est_err_m"] for x in r])), 4)))

    with open(out / "eval_summary.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        wr.writeheader()
        wr.writerows(summary)

    # optional figure for the project page
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        names = [c["name"] for c in CONDITIONS]
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        data = [[x["mean_nees"] for x in rows if x["condition"] == n] for n in names]
        ax[0].boxplot(data, showfliers=True)
        ax[0].set_xticklabels(names)
        ax[0].axhline(3, color="g", ls="--", label="expected (3 DOF)")
        ax[0].set_ylabel("mean NEES per trial")
        ax[0].set_yscale("log")
        ax[0].legend()
        ax[0].tick_params(axis="x", rotation=45)
        ax[1].bar(names, [s["successes"] / s["trials"] * 100 for s in summary])
        ax[1].set_ylabel("success rate (%)")
        ax[1].set_ylim(0, 105)
        ax[1].tick_params(axis="x", rotation=45)
        fig.tight_layout()
        fig.savefig(out / "eval_summary.png", dpi=150)
        print("figure saved to", out / "eval_summary.png")
    except Exception as err:
        print("(skipped figure:", err, ")")

    print("per-trial rows:", out / "eval_trials.csv", "| summary:", out / "eval_summary.csv")