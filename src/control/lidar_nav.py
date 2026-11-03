import numpy as np


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class LidarNavigator:
    """Gap follower: picks a safe heading from the LiDAR, returns a reference for the LQR."""

    def __init__(self, r_safe=0.22, d_look=1.2, lookahead=0.30, span_deg=100, n_cand=41,
                 w_clear=1.5, w_switch=0.3, dock_mask=0.45, d_stop=0.45):
        self.r_safe = r_safe              # robot half-width + margin (m); Waffle Pi is ~0.15 m
        self.d_look = d_look              # how far ahead a candidate heading is checked (m)
        self.lookahead = lookahead        # distance of the reference point ahead of the robot (m)
        self.w_clear, self.w_switch = w_clear, w_switch
        self.dock_mask = dock_mask        # ignore LiDAR points this close to the dock (m)
        self.d_stop = d_stop              # headings with less free space than this are never chosen
        self.phis = np.deg2rad(np.linspace(-span_deg, span_deg, n_cand))   # candidates, robot frame
        self.prev_phi = 0.0

    def plan(self, x_hat, goal, angles, ranges, dock=None, v_max=0.15):
        x, y, th = x_hat
        gx, gy = goal[0] - x, goal[1] - y
        dist = float(np.hypot(gx, gy))
        goal_phi = _wrap(np.arctan2(gy, gx) - th)

        # LiDAR hits in the robot frame
        ok = np.isfinite(ranges)
        a, r = angles[ok], ranges[ok]
        px, py = r * np.cos(a), r * np.sin(a)

        # don't treat the dock as an obstacle
        if dock is not None and px.size:
            wx = x + np.cos(th) * px - np.sin(th) * py
            wy = y + np.sin(th) * px + np.cos(th) * py
            keep = np.hypot(wx - dock[0], wy - dock[1]) > self.dock_mask
            px, py = px[keep], py[keep]

        d_eff = min(self.d_look, dist + self.r_safe)     # don't care about things beyond the goal
        if px.size:
            c, s = np.cos(self.phis)[:, None], np.sin(self.phis)[:, None]
            along = c * px[None, :] + s * py[None, :]            # distance along each candidate ray
            lat = np.abs(-s * px[None, :] + c * py[None, :])     # sideways distance from that ray
            blocked = (along > 0) & (along < d_eff) & (lat < self.r_safe)
            free = np.where(blocked, along, d_eff).min(axis=1)
            min_range = float(np.hypot(px, py).min())
        else:
            free = np.full(self.phis.shape, d_eff)
            min_range = float("inf")

        # score headings: towards goal, open, not switching
        cost = (np.abs(_wrap(self.phis - goal_phi))
                + self.w_clear * (1.0 - free / d_eff)
                + self.w_switch * np.abs(self.phis - self.prev_phi))

        feasible = free >= min(self.d_stop, d_eff)
        if feasible.any():
            i = int(np.argmin(np.where(feasible, cost, np.inf)))
            stuck = False
        else:
            i = int(np.argmax(free))          # everything nearby is blocked: turn toward the most open way
            stuck = True
        phi = float(self.phis[i])
        self.prev_phi = phi

        # slow down near obstacles and the goal
        clear = 1.0 if free[i] >= d_eff else float(np.clip((free[i] - 0.30) / 0.50, 0.0, 1.0))
        v_cap = min(v_max, 0.8 * dist) * clear * max(0.0, np.cos(phi))
        if stuck:
            v_cap = 0.0                       # rotate in place until a heading opens up

        # local reference point on the chosen heading, for the LQR tracker
        psi = th + phi
        L = min(self.lookahead, max(dist, 0.05))
        ref = np.array([x + L * np.cos(psi), y + L * np.sin(psi), psi, max(v_cap, 0.05), 0.0])
        return ref, v_cap, {"phi": phi, "goal_phi": goal_phi, "free": float(free[i]),
                            "min_range": min_range, "dist": dist, "stuck": stuck}