"""Real TurtleBot, same methods as Environment, using the UTS rosbridge library.
  pip install "git+https://github.com/UTS-RI/turtlebot_rosbridge.git"
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from src.hw import hw_config as hw
from src.control.lqr_tracker import R_WHEEL, TRACK

ROOT = Path(__file__).resolve().parents[2]
N_BEAMS = 360


# ---------------- helpers ----------------
def wheels_to_twist(left, right):
    v = R_WHEEL * (left + right) / 2
    w = R_WHEEL * (right - left) / TRACK
    s = hw.SAFETY_SCALE
    return (float(np.clip(v * s, -hw.MAX_LINEAR, hw.MAX_LINEAR)),
            float(np.clip(w * s, -hw.MAX_ANGULAR, hw.MAX_ANGULAR)))


def scan_to_robot_frame(angle_min, angle_increment, ranges, range_min, range_max, lidar_forward):
    r = np.asarray(ranges, float)
    a = angle_min + angle_increment * np.arange(len(r))
    ok = np.isfinite(r) & (r >= max(range_min, 0.01)) & (r <= range_max)
    x = r[ok] * np.cos(a[ok]) + lidar_forward          # move from the LiDAR to the axle
    y = r[ok] * np.sin(a[ok])
    ang = np.arctan2(y, x)
    dist = np.hypot(x, y)
    out = np.full(N_BEAMS, np.inf)
    idx = (np.round(np.degrees(ang)).astype(int) + 180) % N_BEAMS
    np.minimum.at(out, idx, dist)                       # closest hit in each 1 deg bin
    angles = np.linspace(-np.pi, np.pi, N_BEAMS, endpoint=False)
    return angles, out


def robot_ip_from_args(default=hw.ROBOT_IP):
    for arg in sys.argv:
        if arg.startswith("--ip="):
            return arg.split("=", 1)[1]
    return default


# ---------------- the robot ----------------
class RealRobot:

    def __init__(self, ip=None, wait_for_sensors=10.0):
        from rosbridge_api import RosbridgeRobot

        self.ip = ip or robot_ip_from_args()
        print(f"connecting to the robot at {self.ip}:{hw.ROBOT_PORT} ...")
        self.robot = RosbridgeRobot(self.ip, port=hw.ROBOT_PORT)
        self.robot.setup()
        self._last_cmd = (0.0, 0.0)

        # for the stale sensor check
        self._joints_seen, self._t_wheels = None, 0.0
        self._scan_seq, self._t_scan = -1, 0.0
        self._scan_cache = None

        # camera calibration from tools/calibrate_camera_real.py
        cal = ROOT / hw.CAMERA_CALIBRATION_FILE
        if cal.exists():
            c = np.load(cal)
            self._K, self._dist = c["K"], c["dist"].ravel()
        else:
            self._K, self._dist = None, np.zeros(5)
            print(f"WARNING: {hw.CAMERA_CALIBRATION_FILE} not found. Run tools/calibrate_camera_real.py first.")

        self._t0 = time.monotonic()
        self._next_tick = self._t0
        self._wait_for_sensors(wait_for_sensors)
        print("RealRobot ready")

    def _wait_for_sensors(self, timeout):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if (self._read_wheels() is not None and self.robot.query_laser_scan(timeout=0) is not None
                    and self.robot.query_image(timeout=0) is not None):
                return
            time.sleep(0.1)
        missing = [n for n, ok in (("wheels", self._read_wheels() is not None),
                                   ("lidar", self.robot.query_laser_scan(timeout=0) is not None),
                                   ("camera", self.robot.query_image(timeout=0) is not None)) if not ok]
        self.close()
        raise RuntimeError(f"No data from: {', '.join(missing)}. Is the robot on, are you on its Wi-Fi, "
                           f"and is the IP ({self.ip}) right?")

    def _read_wheels(self):
        msg = self.robot.query_joint_states()
        if not msg:
            return None
        if msg is not self._joints_seen:                # a new message arrived
            self._joints_seen, self._t_wheels = msg, time.monotonic()
        try:
            names, pos = list(msg["name"]), list(msg["position"])
            return pos[names.index(hw.WHEEL_JOINTS[0])], pos[names.index(hw.WHEEL_JOINTS[1])]
        except (KeyError, ValueError):
            return None

    # ---------- simulation-style methods ----------
    def reset(self, *_ignored):
        self.set_wheel_speeds(0.0, 0.0)
        self._t0 = time.monotonic()
        self._next_tick = self._t0

    def step(self):
        # resend the last command, stop if a sensor goes quiet
        now = time.monotonic()
        self._read_wheels()
        seq = getattr(self.robot, "_laser_scan_sequence", None)
        if seq is not None and seq != self._scan_seq:
            self._scan_seq, self._t_scan = seq, now
        stale = [n for n, t in (("wheels", self._t_wheels), ("lidar", self._t_scan if seq is not None else now))
                 if now - t > hw.STALE_AFTER]
        if stale:
            self.robot.send_command(0.0, 0.0)
            print(f"WARNING: no recent {', '.join(stale)} data, robot stopped")
        else:
            self.robot.send_command(*self._last_cmd)
        self._next_tick += hw.CONTROL_DT
        time.sleep(max(0.0, self._next_tick - time.monotonic()))
        if time.monotonic() - self._next_tick > 0.2:      # fell well behind: don't try to catch up
            self._next_tick = time.monotonic()

    @property
    def time(self):
        return time.monotonic() - self._t0

    # ---------- actuation ----------
    def set_wheel_speeds(self, left, right):
        self._last_cmd = wheels_to_twist(left, right)
        self.robot.send_command(*self._last_cmd)

    def stop(self):
        self._last_cmd = (0.0, 0.0)
        for _ in range(5):                      # send a few times in case one is lost over Wi-Fi
            self.robot.stop()
            time.sleep(0.02)

    def close(self):
        try:
            if self.robot.is_connected:
                self.stop()
        finally:
            self.robot.cleanup()                # also sends a stop and disconnects

    # ---------- sensors ----------
    def get_wheel_angles(self):
        return self._read_wheels()

    def get_camera_image(self):
        # library gives BGR, perception wants RGB
        bgr = self.robot.query_image(timeout=1.0)
        return None if bgr is None else cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def get_camera_intrinsics(self):
        if self._K is None:
            raise RuntimeError("No camera calibration. Run tools/calibrate_camera_real.py first.")
        return self._K

    def get_camera_distortion(self):
        return self._dist

    def get_camera_mount(self):
        return hw.CAM_FORWARD, hw.CAM_LEFT

    def get_lidar(self):
        scan = self.robot.query_laser_scan(timeout=1.0)
        if scan is None:
            return self._scan_cache
        self._scan_cache = scan_to_robot_frame(scan.angle_min, scan.angle_increment, scan.ranges,
                                               scan.range_min, scan.range_max, hw.LIDAR_FORWARD)
        return self._scan_cache

    # ---------- things a real robot doesn't have ----------
    def get_ground_truth(self):
        raise RuntimeError("A real robot has no ground truth. Measure the final position with a tape measure.")

    def get_ground_truth_marker_frame(self):
        return self.get_ground_truth()

    def get_dock_position(self):
        raise RuntimeError("The robot finds the marker itself; there is no map on the real robot.")
