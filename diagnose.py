import csv
import numpy as np
import evaluate as ev          # reuses its setup; evaluate's main block doesn't run on import

COND = "kw5e-3"                # which condition to replay
MAX_TRIALS = 2                 # how many failing trials to replay

env = ev.Environment(seed=0)
cond = next(c for c in ev.CONDITIONS if c["name"] == COND)
rows = list(csv.DictReader(open("results/eval_trials.csv")))
bad = sorted({int(r["trial"]) for r in rows if r["condition"] == COND and r["failure_cause"] != "ok"})
print("failing trials in", COND, ":", bad)


def trace(trial):
    rng = np.random.default_rng(trial)
    start = np.array([rng.uniform(0.0, 0.3), rng.uniform(-0.3, 0.3), rng.uniform(-0.35, 0.35)])
    dock = env.get_dock_position()
    layout = ev.sample_layout(env, rng, start[:2], dock)
    for g, p in layout.items():
        env.m.geom_pos[g][:2] = p

    ev.config.CAMERA_PIXEL_NOISE = cond["pixel_noise"]
    ev.env_config.ENCODER_NOISE = cond["enc_noise"]
    ev.env_config.LIDAR_NOISE = cond["lidar_noise"]
    env.rng = np.random.default_rng(10_000 + trial)
    cam_rng = np.random.default_rng(20_000 + trial)
    env.reset(*start)
    K_cam = env.get_camera_intrinsics()

    cfg = ev.EKF_build(wheel_radius=0.033, wheel_base=0.288, k_wheel=cond["k_wheel"])
    init_err = np.array([0.03, 0.03, 0.04]) * cond["init_scale"]
    ekf = ev.EKF(start + rng.normal(0, init_err), cfg, P0=cfg.P0 * cond["p0_scale"] ** 2)
    ekf.step(*env.get_wheel_angles())

    u = start[:2] - dock
    u /= np.linalg.norm(u)
    standoff_pt, stop_pt = dock + ev.STANDOFF * u, dock + ev.STOP_DIST * u
    nav = ev.LidarNavigator()

    # ---- harness-only: how much room is there around the stand-off point? ----
    print(f"start=({start[0]:.2f},{start[1]:.2f}) stand-off point=({standoff_pt[0]:.2f},{standoff_pt[1]:.2f})")
    for g, p in layout.items():
        sz = env.m.geom_size[g]
        rad = np.hypot(*sz[:2]) if env.m.geom_type[g] == 6 else sz[0]
        gap = np.linalg.norm(p - standoff_pt) - rad
        flag = "   <-- inside planner safety margin (r_safe=0.22)" if gap < 0.22 else ""
        print(f"   obstacle at ({p[0]:.2f},{p[1]:.2f}) clearance to stand-off point {gap:.2f} m{flag}")

    phase, goal = 0, standoff_pt
    angles = ranges = None
    v_hist, phi_hist = [], []
    k = 0
    while env.time < ev.T_MAX:
        if k % ev.lidar_every == 0:
            angles, ranges = env.get_lidar()
        det = ev.detect_marker(env.get_camera_image(), K_cam, rng=cam_rng) if k % ev.cam_every == 0 else None
        R_meas = ev.measurement_covariance(det[0]) if det is not None else None
        x_hat, P = ekf.step(*env.get_wheel_angles(), det, dock, R_meas)

        if phase == 0 and np.hypot(*(goal - x_hat[:2])) < 0.10:
            phase, goal = 1, stop_pt
        if phase == 1 and np.hypot(*(stop_pt - x_hat[:2])) < 0.04:
            print("   docked at t =", round(env.time, 1))
            return
        ref, v_cap, info = nav.plan(x_hat, goal, angles, ranges, dock, v_max=0.15 if phase == 0 else 0.08)
        v, w = ev.control(x_hat, ref, ev.Q, ev.Rw, ev.dt)
        v = float(np.clip(v, 0.0, v_cap))
        wl, wr = ev.to_wheels(v, w)
        env.set_wheel_speeds(wl, wr)
        env.step()
        v_hist.append(v)
        phi_hist.append(info["phi"])
        gt = env.get_ground_truth()
        if k % 500 == 0:
            print(f"   t={env.time:5.1f} ph={phase} est=({x_hat[0]:.2f},{x_hat[1]:.2f}) TRUE=({gt[0]:.2f},{gt[1]:.2f}) "
                  f"goal_dist={info['dist']:.2f} phi={np.degrees(info['phi']):4.0f} "
                  f"free={info['free']:.2f} v={v:.3f}")
        k += 1

    tail = int(50 / ev.dt)
    v_t, p_t = np.array(v_hist[-tail:]), np.array(phi_hist[-tail:])
    nz = p_t[np.abs(p_t) > 1e-6]
    flips = int(np.sum(np.diff(np.sign(nz)) != 0)) if nz.size > 1 else 0
    print(f"   TIMEOUT. last 50 s: v<0.02 for {100*np.mean(v_t < 0.02):.0f}% of the time, "
          f"chosen heading changed sign {flips} times, final goal_dist {info['dist']:.2f} m, phase {phase}")
    print("   -> mostly stopped + goal inside margin = blocked goal | mostly stopped + many flips = oscillation | "
          "moving but not arriving = path/tracking")


for t in bad[:MAX_TRIALS]:
    print("\n=== trial", t)
    trace(t)