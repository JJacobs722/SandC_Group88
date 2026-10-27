# Running on the real TurtleBot3 Waffle Pi

The same pipeline runs on the real robot. Only the robot class changes:

```
python run_dock.py          # simulation (src/sim/environment.py)
python run_dock.py --real   # real robot (src/hw/real_robot.py)
```

Perception, the EKF, the search, LiDAR navigation and LQR are exactly the same code. `RealRobot` has the
same methods as `Environment`, but talks to the robot over Wi-Fi using the UTS library
[UTS-RI/turtlebot_rosbridge](https://github.com/UTS-RI/turtlebot_rosbridge) (`RosbridgeRobot`) instead of MuJoCo.

| Method | Simulation | Real robot (`RosbridgeRobot`) |
| --- | --- | --- |
| `set_wheel_speeds(l, r)` | MuJoCo motors | `send_command(forward, turn)` on `/cmd_vel` |
| `get_wheel_angles()` | MuJoCo wheel joints | `query_joint_states()` from `/joint_states` |
| `get_lidar()` | 360 MuJoCo rays | `query_laser_scan()` from `/scan`, re-centred on the wheel axle, 360 x 1 deg |
| `get_camera_image()` | MuJoCo render | `query_image()` from `/camera/image_raw/compressed` (BGR, converted to RGB) |
| `get_camera_intrinsics()` | from the 60 deg field of view | from `camera_calibration_real.npz` |
| `get_ground_truth()` | MuJoCo | doesn't exist: measure with a tape |

## What you need

- **Any laptop** (Windows, Mac or Linux). No ROS needed: the robot runs ROS 2 Humble and a rosbridge server,
  and the laptop just connects over Wi-Fi.
- In this repo's venv, install the UTS library once:
  `pip install "git+https://github.com/UTS-RI/turtlebot_rosbridge.git"` (or clone it and `pip install .`)
- Join the robot's Wi-Fi network (name and password are in the UTS-RI/turtlebot_rosbridge README).
- Your robot's IP: `192.168.1.2XX`, where XX is the TurtleBot's number. Put it in `ROBOT_IP` in
  `src/hw/hw_config.py`, or add `--ip=192.168.1.2XX` to any command.
- Sign out the robot and a battery (ask at the lab).
- A printed ArUco marker, taped flat to a wall at camera height.
  **Print `aruco_0_print.png`** from `python -m tools.make_marker_print`, not `aruco_0.png`: that one is
  mirrored on purpose for MuJoCo and won't be detected on paper.
- A tape measure, masking tape, the checkerboard printout (`assets/robotis_tb3/checkerboard.png`).

## Steps at the lab (in order)

1. **Check the connection with the UTS examples** (from the turtlebot_rosbridge folder):
   `python examples/camera.py 192.168.1.2XX`, `python examples/laser_scan.py 192.168.1.2XX`, and
   `python examples/teleop.py 192.168.1.2XX` (W/S/A/D, Space stops). If these work, ours will.
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
| `No module named 'rosbridge_api'` | UTS library not installed in this venv | `pip install "git+https://github.com/UTS-RI/turtlebot_rosbridge.git"` |
| `Could not connect to rosbridge` | Not on the robot's Wi-Fi, wrong IP, or robot still booting | check the Wi-Fi and IP; try `examples/camera.py` first |
| `No data from: wheels` | Wheel joint names differ | print `query_joint_states()` and update `WHEEL_JOINTS` in `hw_config.py` |
| Marker never seen | Printed the flipped marker, bad lighting, or no calibration file | print the unflipped one; run `calibrate_camera_real` |
| Marker distance wrong | `MARKER_SIZE` or calibration wrong | re-measure the black square; recalibrate |
| Robot stops and starts | "no recent ... data" warnings: Wi-Fi lag | move closer to the robot; raise `STALE_AFTER` a little |
| Docks but off to one side | `CAM_LEFT` not measured (Waffle Pi cameras can sit off-centre) | measure it and update `hw_config.py` |

## How this was tested before lab day

There's no robot in the simulation, so `RealRobot` was tested with the UTS `RosbridgeRobot` library running
unchanged, connected to a fake rosbridge that was really the MuJoCo sim sending the same messages the robot
sends (base64 JPEG camera frames, a 0..360 deg LaserScan at 5 Hz with 0 meaning "no return",
`/joint_states`, Twist commands). The full `run_dock.py --real` pipeline searched, found the marker and
docked 2.3 cm from the target through it. `tools/hw_check.py --move` ran the same way. The real robot will
still need the measurements above.
