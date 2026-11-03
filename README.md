# Group 88: TurtleBot3 Visual Docking (Track A)
41014 Sensors and Control for Mechatronic Systems, Integrated Project, Spring 2026

A TurtleBot3 Waffle Pi starts somewhere in a room it has no map of, searches for an ArUco-marked charging station, drives around obstacles with its LiDAR, and docks in front of the marker. An EKF fuses wheel-encoder odometry with the marker's range, bearing and facing from the camera. An LQR tracker drives the robot using only that estimate. Ground truth is never used for control. The same code also runs on the real robot.

| | |
|---|---|
| **Team** | Helena Karedis · Joshua Jacobs · Sebastian Todd · Taara Kotibhaskar |
| **Repository** | https://github.com/JJacobs722/SandC_Group88 |
| **Demo video (5 min)** | [[LINK TO VIDEO]] |

## 1. Install and run
```
python -m venv .venv
.venv\Scripts\activate          # Mac: source .venv/bin/activate
pip install -r requirements.txt
```
| Command | What it does |
|---|---|
| `python run_dock.py` | Full simulation, with the estimate and its uncertainty ellipse drawn live |
| `python run_dashboard.py` | Same run with a 4-panel view (top-down, camera, LiDAR, live numbers) |
| `python evaluate.py` | 20 trials for each of 9 conditions, results saved in `results/` |
| `python run_dock.py --real --ip=192.168.1.2XX` | Same pipeline on the real TurtleBot (setup in `HARDWARE.md`) |

## 2. Code structure
| Folder | What's in it | Lead |
|---|---|---|
| `src/sim/` | MuJoCo world, robot, sensors and noise | Joshua |
| `src/perception/` | Marker detection and its noise model | Helena |
| `src/estimation/` | EKF | Sebastian |
| `src/control/` | LQR, LiDAR navigation, marker search | Taara |
| `src/hw/` | Real-robot version of the environment | Joshua, Helena |

Ground truth only appears in the blocks marked `evaluation only` in `run_dock.py` and `evaluate.py`.

## 3. Model
- **State:** x = [x, y, θ], the middle of the wheel axle, in the marker frame.
- **Process model:** differential drive from the encoder increments, Δs = r(Δφ_r + Δφ_l)/2 and Δθ = r(Δφ_r − Δφ_l)/b. Assumes a flat floor and no slip beyond the calibrated noise.
- **Measurement:** z = [range, bearing, facing] of the marker, which sits at (0, 0) facing +x.
- **Controllability:** the LQR tracking-error model has full rank (3) while the reference is moving and drops to rank 2 when it stops, so the tracker never lets the reference speed reach exactly zero.
- **Observability:** wheels alone can't fix where the robot is, so drift grows. One marker sighting gives the full pose, which is how the EKF starts. When the marker is out of view, the uncertainty grows until the next sighting.

## 4. Sensing, calibration and frames
**Frames**
- **Camera frame:** x right, y down, z forward. 0.07 m in front of the wheel axle (0.073 m on the real robot).
- **LiDAR frame:** 0.064 m behind the wheel axle (0.065 m on the real robot).
- **Robot frame:** x forward, y left, centred on the wheel axle.
- **Marker frame:** marker at (0, 0), face along +x. The EKF works in this frame.
- **Start frame:** where the robot was switched on. Used with the wheels only, before the marker is seen.


| Calibration | Sim | Real robot |
|---|---|---|
| Camera focal length (checkerboard) | 415.1 px (true 415.7), error 0.11 px | 1311 px, error 0.14 px |
| Wheel radius | 0.0281 m effective (spec 0.033) | 0.0327 m |
| Wheel base | 0.351 m effective (spec 0.288) | [[0.287 spec / measured?]] |

Camera noise was measured from 200 readings at each distance. Range noise grows with distance² and bearing noise stays about constant. Facing noise gets large past 2 m, so the EKF only starts once the robot is within 1.8 m.

## 5. Estimator
| Parameter | Value | Why |
|---|---|---|
| Wheel noise k_wheel | 5e-3 | Most consistent in the sweep below |
| Turning slip k_turn | 5e-4 | Measured: the Waffle shuffles when it spins |
| Camera noise R | from the measured noise model | Bigger further away |
| Outlier gate | χ² 99% (11.34) | Rejects bad detections |
| Restart | after 5 rejects in a row | Recovers if the estimate drifts too far |

## 6. Controller
| Parameter | Value | Why |
|---|---|---|
| LQR Q | diag(20, 20, 5) | Position matters most for docking |
| LQR R | diag(1, 2) | Turning penalised more, to limit slip |
| Obstacle avoidance | 41 headings, 0.22 m safety radius | Picks open directions from the LiDAR |
| Docking | 0.60 m stand-off, stop at 0.25 m | Lines up square before the final approach |
| Speed | 0.15 m/s, 0.08 m/s on the final approach | Slower when close, for accuracy |

## 7. Results
**Simulation:** 20 trials per condition, random obstacle layouts.

| Condition | Docked | NEES (ideal 3) |
|---|---|---|
| Nominal | 16/20 | 2.57 |
| Camera noise ×4 | 16/20 | 7.71 |
| LiDAR noise ×5 | 13/20 | 3.42 |
| Encoder noise ×10 | 16/20 | 5.20 |
| Bad start guess ×6 | 16/20 | 12.53 |
| Bad start guess ×6, wider start uncertainty | 17/20 | 4.12 |

- Nominal docks finished **4.7 cm** from the target on average (worst 5.1 cm), with no collisions.
- **Trade-off:** k_wheel 1e-3 gave NEES 6.47 (overconfident) and 2e-2 gave 1.44 (underconfident). 5e-3 was the best balance.

**Real robot:** 12 runs with a 15 cm marker.

| | |
|---|---|
| Docked | 11/12 (the 1 failure never found the marker) |
| Had to search first | 7 of 12 runs |
| Time to dock | 15–84 s |


The real camera and wheels both needed re-calibrating. The estimate's heading uncertainty also grew near the end of the real runs, because the marker leaves the camera's view in the last 25 cm.

## 8. Limitations
- The search can miss the marker (1 real run).
- The final approach is blind once the marker is too close to see.


## 9. Individual contributions
| Member | Contribution |
|---|---|
| Joshua Jacobs | Simulated world (robot, room, sensors, noise), wheel and camera calibration, real-robot interface, real TurtleBot testing |
| Helena Karedis | Perception (marker detection, noise model), real-robot interface, real TurtleBot testing  |
| Sebastian Todd | Estimation (EKF design and tuning)  |
| Taara Kotibhaskar | Control (LQR, LiDAR navigation, docking), demo video  |

## 10. Generative AI declaration
| Tool | Used for |
|---|---|
| Claude (Anthropic) | Debuggin, integration, evaluation scripts, real-robot interface and documentation. |

## 11. Acknowledgements
TurtleBot3 model: ROBOTIS mujoco menagerie (Apache 2.0) · Real-robot interface: UTS-RI/turtlebot_rosbridge · EKF structure adapted from the 41014 tutorial / Assessment 2 code
