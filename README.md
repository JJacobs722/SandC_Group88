# SandC_Group88: TurtleBot3 Visual Docking (Track A)

41014 Sensors and Control, Integrated Project. A TurtleBot3 Waffle Pi in MuJoCo navigates
a room with obstacles and docks at an ArUco-marked charging station, using an EKF on
wheel odometry and camera measurements, and an LQR controller running on the estimated state.

## Setup

```
python -m venv .venv
.venv\Scripts\activate          # Mac: source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```
python main.py                    # demo
python -m tools.test_sensors      # check camera and LiDAR
python -m tools.calibrate_wheels  # measure wheel radius and track width
```

## Project structure

```
assets/robotis_tb3/   robot model, world.xml (room, obstacles, dock), ArUco marker
src/sim/              environment: the only code that talks to MuJoCo
src/perception/       marker detection
src/estimation/       EKF
src/control/          controller
tools/                test and calibration scripts
```

## Environment interface (`src/sim/environment.py`)

| Method | Returns | Units / frame |
|---|---|---|
| `reset(x, y, yaw)` | none | m, rad, world frame |
| `step()` | none | advances 0.02 s (50 Hz) |
| `set_wheel_speeds(left, right)` | none | rad/s, clipped to ±7.88 |
| `get_wheel_angles()` | `(left, right)` | rad, cumulative, with noise |
| `get_camera_image()` | RGB array (480, 640, 3) | uint8 |
| `get_camera_intrinsics()` | 3×3 K matrix | pixels |
| `get_lidar()` | `(angles, ranges)` | rad in robot frame, m; no hit = inf |
| `get_dock_position()` | `(x, y)` | m, world frame (known map) |
| `get_ground_truth()` | `(x, y, yaw)` | **evaluation only, never used by the estimator or controller** |

## Frames

- **World:** origin at the room centre, x towards the dock, z up. Yaw is measured anticlockwise from +x.
- **Robot:** x forward, y left. LiDAR angle 0 is straight ahead; positive angles are to the left.
- **Camera:** looks along the robot's x-axis, mounted 0.07 m forward and 0.12 m up.

## Robot parameters

| Parameter | Value | Source |
|---|---|---|
| Wheel radius (effective) | **0.0281 m** | calibrated, `tools/calibrate_wheels.py` |
| Track width (effective) | **0.351 m** | calibrated |
| Wheel radius (geometric) | 0.033 m | model mesh |
| Track width (geometric) | 0.288 m | model wheel positions |
| Max wheel speed | ±7.88 rad/s | model actuator range |
| Physics timestep | 0.002 s | model |
| Control rate | 50 Hz | `config.py` |

Use the **effective** values for odometry. The robot travels about 15% less than its geometric wheel
size predicts (tyre slip and soft contact), and turns less than its geometric track width predicts
(tyre scrub when turning).

**Motor gain:** the wheel actuators' `kv` was raised to 1 in `turtlebot3_waffle_pi.xml`. With the original
gain, the wheels only reached about 60% of the commanded speed. The real TurtleBot3's Dynamixel motors
have built-in velocity control that tracks accurately, so the higher gain matches the real robot.
At kv = 1, the wheels reach about 96% of a 5 rad/s command within 5 s from a standstill.

## Sensors

| Sensor | Setting |
|---|---|
| Camera | 640×480, vertical FOV 60°, K: f = 415.7, cx = 320, cy = 240 |
| LiDAR | 360 beams (1°), max range 3.5 m, noise σ = 0.01 m |
| Wheel encoders | noise σ = 0.002 rad |
| ArUco marker | `DICT_4X4_50`, ID 0, 0.20 m black square, centre at (2.44, 0, 0.15) |

All settings live in `src/sim/config.py`.

**Marker note:** `aruco_0.png` is saved mirrored on purpose, because MuJoCo mirrors it again on the box face.
Keep the `cv2.flip` line in `tools/make_marker.py` if you regenerate it.

## World

5 m × 5 m room, walls 0.5 m tall. The robot starts at (0, 0) facing the dock. `obstacle_1` sits directly
in the path, so the marker is hidden at the start and the robot must drive around it.
Obstacle positions can be changed in `world.xml`.

## Team contributions

Josh - Simulated world

Helena - Perception

Seb - Estimation

Taara - Control

## Generative AI declaration

Claude (Anthropic) was used to help set up the simulation environment and write README.md

## Acknowledgements

- TurtleBot3 model: [ROBOTIS robotis_mujoco_menagerie](https://github.com/ROBOTIS-GIT/robotis_mujoco_menagerie), Apache 2.0
- MuJoCo, OpenCV