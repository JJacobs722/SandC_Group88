# Running on the real TurtleBot3 Waffle Pi

The same pipeline runs on the real robot. Only the robot class changes:

```
python run_dock.py          # simulation (src/sim/environment.py)
python run_dock.py --real   # real robot (src/hw/real_robot.py)
```

Perception, the EKF, the search, LiDAR navigation and LQR are exactly the same code. `RealRobot` has the
same methods as `Environment`, but talks to the robot over ROS 2 instead of MuJoCo.

| Method | Simulation | Real robot |
| --- | --- | --- |
| `set_wheel_speeds(l, r)` | MuJoCo motors | `/cmd_vel` (converted to forward speed + turn rate) |
| `get_wheel_angles()` | MuJoCo wheel joints | `/joint_states` |
| `get_lidar()` | 360 MuJoCo rays | `/scan`, re-centred on the wheel axle, 360 x 1 deg |
| `get_camera_image()` | MuJoCo render | camera topic (raw or compressed) |
| `get_camera_intrinsics()` | from the 60 deg field of view | from `camera_calibration_real.npz` |
| `get_ground_truth()` | MuJoCo | doesn't exist: measure with a tape |

## What you need

- A computer with **ROS 2** and `rclpy` (usually the lab's Ubuntu machines). `rclpy` can't be pip-installed.
  Then in this repo: `pip install numpy scipy opencv-python matplotlib`. MuJoCo isn't needed for `--real`.
- The robot powered on, on the same network, running its bringup and camera node
  (the lab will show you, usually `ros2 launch turtlebot3_bringup robot.launch.py`).
- The same `ROS_DOMAIN_ID` on the laptop as the robot (ask at the induction).
- A printed ArUco marker: `assets/robotis_tb3/aruco_0.png` (ID 0, DICT_4X4_50), taped flat to a wall at camera height.
  **Print `aruco_0_print.png`** from `python -m tools.make_marker_print`, not `aruco_0.png`: that one is
  mirrored on purpose for MuJoCo and won't be detected on paper.
- A tape measure, masking tape, the checkerboard printout (`assets/robotis_tb3/checkerboard.png`).

## Steps at the lab (in order)

1. **Check the topics.** `ros2 topic list` should show `/cmd_vel`, `/joint_states`, `/scan` and a camera topic.
   If any names differ, change them in `src/hw/hw_config.py`.
2. **Sensor check, robot still:** `python -m tools.hw_check`. Look at `hw_camera.png` and `hw_lidar.png`.
3. **Motor check:** `python -m tools.hw_check --move`. It drives about 10 cm then turns left. If it turns
   right, the wheel names in `WHEEL_JOINTS` are swapped.
4. **Measure by hand** and put the numbers in `src/hw/hw_config.py`:
   `CAM_FORWARD`, `CAM_LEFT` (camera lens from the middle of the wheel axle), `LIDAR_FORWARD`, `MARKER_SIZE`
   (black square only).
5. **Calibrate the camera:** `python -m tools.calibrate_camera_real` (instructions at the top of the file).
   Saves `camera_calibration_real.npz`. Commit it.
6. **Calibrate the wheels:** `python -m tools.calibrate_wheels_real`. Paste `WHEEL_RADIUS` and `WHEEL_BASE`
   into `hw_config.py`.
7. **Run it:** put the robot down anywhere with the marker somewhere in the room, then `python run_dock.py --real`.
   **Ctrl+C stops the robot.** Every run saves `results/real_run_<time>.csv` (what the robot believed, tick by tick).
8. **Record the result:** tape-measure the robot's final distance from the marker (target 25 cm, straight in front).
   Repeat from several start spots. That table is the sim-vs-real comparison the brief asks for.

## Safety

- `SAFETY_SCALE = 0.6` in `hw_config.py` makes everything 40% slower. Raise it to 1.0 only once it works.
- The robot stops by itself if the wheels or LiDAR go quiet for 0.5 s, and on Ctrl+C or any error.
- Keep a hand near the power switch for the first runs.

## If something goes wrong

| Problem | Likely cause | Fix |
| --- | --- | --- |
| `No module named 'rclpy'` | ROS 2 not set up in this terminal | `source /opt/ros/<distro>/setup.bash` first |
| `No data from: ...` | Robot bringup not running, wrong topic name or `ROS_DOMAIN_ID` | check `ros2 topic list` and `ros2 topic hz /scan` |
| Robot doesn't move | `/cmd_vel` type mismatch | the code picks Twist or TwistStamped automatically; check `ros2 topic info /cmd_vel` |
| Marker never seen | Printed the flipped marker, bad lighting, or no calibration file | print the unflipped one; run `calibrate_camera_real` |
| Marker distance wrong | `MARKER_SIZE` or calibration wrong | re-measure the black square; recalibrate |
| Robot stops and starts | "no recent lidar data" warnings: Wi-Fi lag | move closer to the router, or run on the robot's own computer |
| Docks but off to one side | `CAM_LEFT` not measured (Waffle Pi cameras can sit off-centre) | measure it and update `hw_config.py` |

## How this was tested before lab day

There's no robot in the simulation, so `RealRobot` was tested against a fake ROS 2 layer that was really the
MuJoCo sim sending real-robot-style messages (JPEG camera frames, a 0..360 deg LaserScan with 0 meaning
"no return", Twist commands, `/joint_states`). The full `run_dock.py --real` pipeline searched, found the
marker and docked 3.0 cm from the target through that layer. `tools/hw_check.py` and
`tools/calibrate_wheels_real.py` were run the same way. The real robot will still need the
measurements above.
