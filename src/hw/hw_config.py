# Settings for the real TurtleBot3 Waffle Pi (sim settings are in src/sim/config.py)

# ---------- connecting ----------
ROBOT_IP = "192.168.1.213"   # or run_dock.py --real --ip=...
ROBOT_PORT = 9090
WHEEL_JOINTS = ("wheel_left_joint", "wheel_right_joint")

# ---------- limits ----------
MAX_LINEAR = 0.26        # m/s
MAX_ANGULAR = 1.82       # rad/s
SAFETY_SCALE = 0.6       # all commands scaled by this, 1.0 = full speed

# ---------- timing ----------
CONTROL_DT = 0.02        # s
STALE_AFTER = 0.6        # s, stop if wheels or LiDAR go quiet this long

# ---------- camera ----------
CAMERA_CALIBRATION_FILE = "camera_calibration_real.npz"
CAM_FORWARD = 0.073      # m, in front of the wheel axle
CAM_LEFT = 0.0           # m

# ---------- LiDAR ----------
LIDAR_FORWARD = -0.065   # m, behind the wheel axle

# ---------- marker ----------
MARKER_SIZE = 0.15       # m, black square only

# ---------- EKF ----------
WHEEL_RADIUS = 0.0327    # m, measured
WHEEL_BASE = 0.287       # m, spec
K_WHEEL = 5e-3
K_TURN = 5e-4            # sim value
