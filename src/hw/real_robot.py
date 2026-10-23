"""The real TurtleBot3 Waffle Pi, with exactly the same methods as src/sim/environment.py.

run_dock.py only ever calls these methods, so swapping the simulator for the real robot is
one line:   env = Environment(seed=0)   ->   env = RealRobot()

Talks to the robot over ROS 2 (needs a machine with ROS 2 and rclpy installed, on the same
network and ROS_DOMAIN_ID as the robot). The robot itself must be running its bringup
(on the robot:  ros2 launch turtlebot3_bringup robot.launch.py  plus the camera node).

  sim method                    real robot
  set_wheel_speeds(l, r)  ->    /cmd_vel   (converted to forward speed + turn rate)
  get_wheel_angles()      ->    /joint_states
  get_lidar()             ->    /scan      (re-centred on the wheel axle, 360 x 1 deg)
  get_camera_image()      ->    camera topic (raw or compressed)
  get_ground_truth()      ->    doesn't exist on a real robot
"""
import math
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from src.hw import hw_config as hw
from src.control.lqr_tracker import R_WHEEL, TRACK   # same numbers the controller used to make the wheel speeds

ROOT = Path(__file__).resolve().parents[2]
N_BEAMS = 360


# ---------------- pure helpers (no ROS needed, tested on their own) ----------------
def wheels_to_twist(left, right):
    """Wheel speeds (rad/s) -> (forward m/s, turn rad/s), clipped to the robot's limits and the safety scale."""
    v = R_WHEEL * (left + right) / 2
    w = R_WHEEL * (right - left) / TRACK
    s = hw.SAFETY_SCALE
    return (float(np.clip(v * s, -hw.MAX_LINEAR, hw.MAX_LINEAR)),
            float(np.clip(w * s, -hw.MAX_ANGULAR, hw.MAX_ANGULAR)))


def scan_to_robot_frame(angle_min, angle_increment, ranges, range_min, range_max, lidar_forward=hw.LIDAR_FORWARD):
    """A LaserScan -> (angles, ranges) like the sim: 360 beams, 1 deg apart, -180..179 deg,
    measured from the wheel axle midpoint (the point the EKF tracks). No return = inf."""
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


def decode_image(msg_format, data, width=None, height=None, encoding=None):
    """ROS image -> RGB uint8 array (H, W, 3), or None if it can't be read."""
    buf = np.frombuffer(bytes(data), np.uint8)
    if msg_format is not None:                          # CompressedImage
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        return None if bgr is None else cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    enc = (encoding or "").lower()
    if enc in ("rgb8", "bgr8"):
        img = buf.reshape(height, width, 3)
        return img.copy() if enc == "rgb8" else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if enc in ("yuyv", "yuv422_yuy2"):
        return cv2.cvtColor(buf.reshape(height, width, 2), cv2.COLOR_YUV2RGB_YUY2)
    if enc == "mono8":
        return cv2.cvtColor(buf.reshape(height, width), cv2.COLOR_GRAY2RGB)
    return None


# ---------------- the robot ----------------
class RealRobot:
    """Same methods as Environment, backed by the real robot over ROS 2."""

    def __init__(self, wait_for_sensors=10.0):
        import rclpy                                     # only needed on the real robot
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data
        from rclpy.executors import SingleThreadedExecutor
        from sensor_msgs.msg import JointState, LaserScan, Image, CompressedImage
        from geometry_msgs.msg import Twist, TwistStamped

        self._rclpy = rclpy
        if not rclpy.ok():
            rclpy.init()
        self.node = Node("sandc_group88")
        self._lock = threading.Lock()
        self._wheels = self._scan = self._image = None
        self._t_wheels = self._t_scan = self._t_image = 0.0
        self._last_cmd = (0.0, 0.0)

        # sensors: best-effort QoS so it works whatever the robot publishes with
        self.node.create_subscription(JointState, hw.JOINT_STATES_TOPIC, self._on_joints, qos_profile_sensor_data)
        self.node.create_subscription(LaserScan, hw.SCAN_TOPIC, self._on_scan, qos_profile_sensor_data)

        # camera: use the first topic in the list that actually exists
        self.camera_topic = None
        known = dict(self._wait_for_topics(hw.CAMERA_TOPICS + [hw.CMD_VEL_TOPIC], 3.0))
        for t in hw.CAMERA_TOPICS:
            if t in known:
                kind = CompressedImage if "Compressed" in known[t][0] else Image
                self.node.create_subscription(kind, t, self._on_image, qos_profile_sensor_data)
                self.camera_topic = t
                break

        # motors: newer TurtleBot3 builds use TwistStamped on /cmd_vel, older ones Twist
        types = known.get(hw.CMD_VEL_TOPIC, ["geometry_msgs/msg/Twist"])
        self._stamped = any("TwistStamped" in t for t in types)
        self._Twist = TwistStamped if self._stamped else Twist
        self._cmd_pub = self.node.create_publisher(self._Twist, hw.CMD_VEL_TOPIC, 10)

        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self.node)
        self._spin = threading.Thread(target=self._executor.spin, daemon=True)
        self._spin.start()

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
        print(f"RealRobot ready. camera: {self.camera_topic}  cmd_vel: {'TwistStamped' if self._stamped else 'Twist'}")

    # ---------- ROS callbacks: just store the latest message ----------
    def _on_joints(self, msg):
        try:
            l = msg.position[msg.name.index(hw.WHEEL_JOINTS[0])]
            r = msg.position[msg.name.index(hw.WHEEL_JOINTS[1])]
        except ValueError:
            return
        with self._lock:
            self._wheels, self._t_wheels = (l, r), time.monotonic()

    def _on_scan(self, msg):
        out = scan_to_robot_frame(msg.angle_min, msg.angle_increment, msg.ranges, msg.range_min, msg.range_max,
                                  hw.LIDAR_FORWARD)
        with self._lock:
            self._scan, self._t_scan = out, time.monotonic()

    def _on_image(self, msg):
        if hasattr(msg, "format"):
            img = decode_image(msg.format, msg.data)
        else:
            img = decode_image(None, msg.data, msg.width, msg.height, msg.encoding)
        if img is not None:
            with self._lock:
                self._image, self._t_image = img, time.monotonic()

    def _wait_for_topics(self, wanted, timeout):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            found = [(n, t) for n, t in self.node.get_topic_names_and_types() if n in wanted]
            if len(found) == len(wanted):
                return found
            time.sleep(0.2)
        return [(n, t) for n, t in self.node.get_topic_names_and_types() if n in wanted]

    def _wait_for_sensors(self, timeout):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            with self._lock:
                ready = self._wheels is not None and self._scan is not None and self._image is not None
            if ready:
                return
            time.sleep(0.1)
        missing = [n for n, v in (("wheels", self._wheels), ("lidar", self._scan), ("camera", self._image)) if v is None]
        raise RuntimeError(f"No data from: {', '.join(missing)}. Is the robot bringup running, and the topics in hw_config.py right?")

    # ---------- simulation-style methods ----------
    def reset(self, *_ignored):
        """Can't teleport a real robot. Just stop it and restart the clock."""
        self.set_wheel_speeds(0.0, 0.0)
        self._t0 = time.monotonic()
        self._next_tick = self._t0

    def step(self):
        """Send the latest command, then wait until the next 0.02 s tick. Stops the robot if a sensor goes quiet."""
        now = time.monotonic()
        with self._lock:
            stale = [n for n, t in (("wheels", self._t_wheels), ("lidar", self._t_scan)) if now - t > hw.STALE_AFTER]
        if stale:
            self._publish(0.0, 0.0)
            print(f"WARNING: no recent {', '.join(stale)} data, robot stopped")
        else:
            self._publish(*self._last_cmd)
        self._next_tick += hw.CONTROL_DT
        time.sleep(max(0.0, self._next_tick - time.monotonic()))
        if time.monotonic() - self._next_tick > 0.2:      # fell well behind: don't try to catch up
            self._next_tick = time.monotonic()

    @property
    def time(self):
        return time.monotonic() - self._t0

    # ---------- actuation ----------
    def set_wheel_speeds(self, left, right):
        """Wheel speeds in rad/s, same as the sim. Sent as forward speed + turn rate on /cmd_vel."""
        self._last_cmd = wheels_to_twist(left, right)
        self._publish(*self._last_cmd)

    def _publish(self, v, w):
        msg = self._Twist()
        t = msg.twist if self._stamped else msg
        if self._stamped:
            msg.header.stamp = self.node.get_clock().now().to_msg()
        t.linear.x, t.angular.z = float(v), float(w)
        self._cmd_pub.publish(msg)

    def stop(self):
        self._last_cmd = (0.0, 0.0)
        for _ in range(5):                      # send a few times in case one is lost over Wi-Fi
            self._publish(0.0, 0.0)
            time.sleep(0.02)

    def close(self):
        self.stop()
        self._executor.shutdown()
        self.node.destroy_node()
        if self._rclpy.ok():
            self._rclpy.shutdown()

    # ---------- sensors ----------
    def get_wheel_angles(self):
        """Encoder readings (rad), left and right. Real encoders, so no added noise."""
        with self._lock:
            return self._wheels

    def get_camera_image(self):
        """Latest camera frame, RGB (H, W, 3) uint8."""
        with self._lock:
            return self._image

    def get_camera_intrinsics(self):
        if self._K is None:
            raise RuntimeError("No camera calibration. Run tools/calibrate_camera_real.py first.")
        return self._K

    def get_camera_distortion(self):
        return self._dist

    def get_camera_mount(self):
        """(forward, left) of the camera from the wheel axle midpoint, in metres. Measured by hand."""
        return hw.CAM_FORWARD, hw.CAM_LEFT

    def get_lidar(self):
        """(angles, ranges) like the sim: 360 beams, robot frame, no hit = inf."""
        with self._lock:
            return self._scan

    # ---------- things a real robot doesn't have ----------
    def get_ground_truth(self):
        raise RuntimeError("A real robot has no ground truth. Measure the final position with a tape measure.")

    def get_ground_truth_marker_frame(self):
        return self.get_ground_truth()

    def get_dock_position(self):
        raise RuntimeError("The robot finds the marker itself; there is no map on the real robot.")
