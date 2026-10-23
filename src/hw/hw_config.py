# Settings for the REAL TurtleBot3 Waffle Pi. Check every line marked CHECK at the lab.
# The simulation settings stay in src/sim/config.py.

# ---------- ROS 2 topics (CHECK with `ros2 topic list` on the lab machine) ----------
CMD_VEL_TOPIC = "/cmd_vel"
JOINT_STATES_TOPIC = "/joint_states"
SCAN_TOPIC = "/scan"
# tried in this order, the first one that exists is used. compressed is lighter over Wi-Fi
CAMERA_TOPICS = ["/camera/image_raw/compressed", "/camera/image_raw", "/image_raw/compressed", "/image_raw"]
WHEEL_JOINTS = ("wheel_left_joint", "wheel_right_joint")   # names in /joint_states (CHECK)

# ---------- robot geometry used to turn wheel speeds into /cmd_vel ----------
# the robot's own firmware turns /cmd_vel back into wheel speeds using its spec numbers,
# so these must match the ones in src/control/lqr_tracker.py (it is imported from there)

# ---------- limits (real Waffle Pi spec) ----------
MAX_LINEAR = 0.26        # m/s
MAX_ANGULAR = 1.82       # rad/s
SAFETY_SCALE = 0.6       # extra safety for first runs: all commands are scaled down by this. 1.0 = full speed

# ---------- timing ----------
CONTROL_DT = 0.02        # s, same tick as the sim
STALE_AFTER = 0.5        # s: if a sensor hasn't updated for this long, stop the robot

# ---------- camera (CHECK: measure on the robot and run tools/calibrate_camera_real.py) ----------
CAMERA_CALIBRATION_FILE = "camera_calibration_real.npz"   # made by tools/calibrate_camera_real.py
CAM_FORWARD = 0.07       # m: camera lens in front of the wheel axle midpoint (measure it)
CAM_LEFT = 0.0           # m: camera lens to the left of the centre line (negative = right). measure it

# ---------- LiDAR mount (CHECK: measure) ----------
LIDAR_FORWARD = -0.064   # m: LiDAR centre relative to the wheel axle (negative = behind). Waffle Pi spec

# ---------- marker (CHECK: measure the printed marker) ----------
MARKER_SIZE = 0.20       # m: width of the black square only

# ---------- EKF numbers for the real robot (update after tools/calibrate_wheels_real.py) ----------
WHEEL_RADIUS = 0.033     # m, spec value until measured
WHEEL_BASE = 0.287       # m, spec value until measured
K_WHEEL = 5e-3
K_TURN = 5e-4            # sim value. re-measure the shuffle on the real floor if you can
