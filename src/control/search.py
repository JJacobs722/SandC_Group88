"""Marker search: spin, explore towards open space, or approach a far marker. Odometry only, start frame."""
import numpy as np


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class MarkerSearch:
    def __init__(self, spin_rate=0.6, step=1.0, approach_dist=1.3, explore_time=12.0, approach_time=25.0):
        self.spin_rate = spin_rate            # rad/s when turning on the spot
        self.step = step                      # m: furthest each explore move goes
        self.approach_dist = approach_dist    # m: stop this far from the marker when approaching
        self.explore_time = explore_time      # s: give up on an explore move after this long
        self.approach_time = approach_time    # s: give up approaching after this long
        self.mode = "spin"
        self.turned = 0.0
        self.prev_th = None
        self.timer = 0.0
        self.target = None
        self.visited = [np.zeros(2)]          # places we've already spun at (start frame)
        self.marker_xy = None                 # marker position in the start frame, once seen

    def saw_marker(self, pose, det):
        x, y, th = pose
        r, b = det[0], det[1]
        self.marker_xy = np.array([x + r * np.cos(th + b), y + r * np.sin(th + b)])
        if self.mode != "approach":
            self.mode, self.timer = "approach", 0.0

    def update(self, pose, angles, ranges, dt):
        self.timer += dt

        if self.mode == "approach":
            d = self.marker_xy - pose[:2]
            dist = float(np.linalg.norm(d))
            if self.timer > self.approach_time:
                self._start_spin(pose)            # couldn't get close, look around again
                return "spin", None
            return "goto", self.marker_xy - d / max(dist, 1e-6) * self.approach_dist

        if self.mode == "spin":
            th = pose[2]
            if self.prev_th is not None:
                self.turned += abs(_wrap(th - self.prev_th))
            self.prev_th = th
            if self.turned < 2 * np.pi:
                return "spin", None
            # full circle and nothing seen: go somewhere new
            self.target = self._pick_target(pose, angles, ranges)
            self.mode, self.timer = "explore", 0.0

        # explore
        if (self.target is None or np.linalg.norm(self.target - pose[:2]) < 0.15
                or self.timer > self.explore_time):
            self._start_spin(pose)
            return "spin", None
        return "goto", self.target

    def _start_spin(self, pose):
        self.visited.append(np.array(pose[:2], float))
        self.mode, self.turned, self.prev_th, self.timer = "spin", 0.0, None, 0.0

    def _pick_target(self, pose, angles, ranges):
        if angles is None:
            return None
        x, y, th = pose
        r = np.where(np.isfinite(ranges), ranges, 3.5)
        n = len(angles)
        best, best_score = None, -np.inf
        for i in range(0, n, 10):                                  # every 10 degrees
            free = r[np.arange(i - 10, i + 11) % n].min()          # closest hit within +-10 deg
            step = min(self.step, free - 0.5)                      # keep 0.5 m clear of it
            if step < 0.3:
                continue
            a = th + angles[i]
            p = np.array([x + step * np.cos(a), y + step * np.sin(a)])
            new = min(min(np.linalg.norm(p - v) for v in self.visited), 1.5)
            score = step + 0.8 * new
            if score > best_score:
                best, best_score = p, score
        return best
