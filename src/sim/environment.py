#Import Setting files

import math
from pathlib import Path

import numpy as np
import mujoco

from src.sim import config

ROOT = Path(__file__).resolve().parents[2] #works out where project folder is on computer


class Environment:
    """The only way the rest of the code talks to the simulator."""

    def __init__(self, seed=None):
        self.m = mujoco.MjModel.from_xml_path(str(ROOT / config.MODEL_FILE)) #Loads the world.xml and stores it as self.m
        self.d = mujoco.MjData(self.m) #Creates a random number generator which is where all the sensor noise comes from 
        self.rng = np.random.default_rng(seed)

        self.steps_per_control = round(config.CONTROL_DT / self.m.opt.timestep) #Works out how many tiny physics steps to fit into one robot "thought"

        #Finding the wheels and motors:
        self._qadr_left = self.m.jnt_qposadr[self.m.joint("wheel_left").id]
        self._qadr_right = self.m.jnt_qposadr[self.m.joint("wheel_right").id]
        self._act_left = self.m.actuator("wheel_left").id
        self._act_right = self.m.actuator("wheel_right").id
        self.max_wheel_speed = self.m.actuator_ctrlrange[self._act_left, 1]

        #Finding the robot's body
        base = self.m.joint("base_joint").id
        self._base_qadr = self.m.jnt_qposadr[base]
        self._base_body = self.m.jnt_bodyid[base]

        #Setting up the LiDAR
        self._lidar_site = self.m.site(config.LIDAR_SITE).id
        self.lidar_angles = np.linspace(-np.pi, np.pi, config.LIDAR_BEAMS, endpoint=False) #makes a list of 360 angles, evenly spread all the way around
        self._ray_dirs = np.stack( #turn each angle into an arrow direction:
            [np.cos(self.lidar_angles), np.sin(self.lidar_angles), np.zeros(config.LIDAR_BEAMS)],
            axis=1,
        )
        self._geomid = np.zeros(1, dtype=np.int32)

        #Create a render that takes photos of the scene: 
        self._renderer = mujoco.Renderer(self.m, config.CAMERA_HEIGHT, config.CAMERA_WIDTH)
        mujoco.mj_forward(self.m, self.d) #tells MuJoCo to calculate where everything is

    # ---------- simulation ----------
    def reset(self, x=0.0, y=0.0, yaw=0.0): #reset --> putting the robot somewhere
        """Put the robot at (x, y) facing yaw (rad), and restart the clock."""
        mujoco.mj_resetData(self.m, self.d)
        a = self._base_qadr #"where the robot's position starts in qpos"
        self.d.qpos[a:a + 2] = [x, y] #sets robot x and y 
        self.d.qpos[a + 3:a + 7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)] #sets which way the robot faces. 
        mujoco.mj_forward(self.m, self.d) #recalculates everything with the robot in its new spot

    def step(self): 
        """Advance one control period (CONTROL_DT seconds):
        Moves time forward by 0.02s"""
        for _ in range(self.steps_per_control): 
            mujoco.mj_step(self.m, self.d) #runs the physics forward one tiny 0.002 step: motors push --> wheels turn --> robot moves

    @property #lets anyone check the simulation clock by writing env.time
    def time(self):
        return self.d.time

    # ---------- actuation ----------
    def set_wheel_speeds(self, left, right): #takes speed for each wheel in rad/s
        """Wheel speeds in rad/s, clipped to the motor limit."""
        lim = self.max_wheel_speed 
        self.d.ctrl[self._act_left] = np.clip(left, -lim, lim) #sends each speed to its motor
        self.d.ctrl[self._act_right] = np.clip(right, -lim, lim)

    # ---------- sensors ----------
    def get_wheel_angles(self): #the odometer
        """Encoder readings (rad) for the left and right wheels, with noise."""
        n = config.ENCODER_NOISE
        left = self.d.qpos[self._qadr_left] + self.rng.normal(0, n) #reads each weel's true angle from qpos, then adds a small random error
        right = self.d.qpos[self._qadr_right] + self.rng.normal(0, n)
        return left, right

    def get_camera_image(self): #the camera
        """RGB image, shape (H, W, 3), uint8."""
        self._renderer.update_scene(self.d, camera=config.CAMERA_NAME) #sets up the shot: the current state of the world viewed from front_cam
        return self._renderer.render() #takes the photo and returns it as array of shapes

    def get_camera_intrinsics(self): #gets the camera's f.o.v and converts it to rads
        """3x3 camera matrix K (pixels)."""
        fovy = math.radians(self.m.cam_fovy[self.m.camera(config.CAMERA_NAME).id])
        f = (config.CAMERA_HEIGHT / 2) / math.tan(fovy / 2) #works out the focal length in pixels --> "how zoomed in camera is"
        return np.array([[f, 0, (config.CAMERA_WIDTH - 1) / 2], #builds K matrix (how the camera is described)
                         [0, f, (config.CAMERA_HEIGHT - 1) / 2],
                         [0, 0, 1]])

    def get_camera_mount(self):
        p = self.m.cam_pos[self.m.camera(config.CAMERA_NAME).id]
        return float(p[0]), float(p[1])

    def get_camera_distortion(self):
        return np.zeros(5)

    def get_lidar(self): #gets where the red dot is in the world rn
        """Returns (angles in rad, ranges in m). No hit = inf. Angles are in the robot frame."""
        origin = self.d.site_xpos[self._lidar_site].copy()
        R = self.d.site_xmat[self._lidar_site].reshape(3, 3) #gets which way red dot is facing
        ranges = np.full(config.LIDAR_BEAMS, np.inf) #makes list of 360 slots (all starting as infinity) --> any beam that hits nothing stays at infinity
        for i, v in enumerate(self._ray_dirs): #goes throuhg all beams:
            dist = mujoco.mj_ray(self.m, self.d, origin, R @ v, None, 1, #fires one laser beam and gets the distance to whatever it hits
                                 self._base_body, self._geomid)
            if 0 <= dist <= config.LIDAR_MAX_RANGE:
                ranges[i] = dist + self.rng.normal(0, config.LIDAR_NOISE)
        return self.lidar_angles, ranges

    # ---------- dock position (evaluate.py and test tools only, not the robot) ----------
    def get_dock_position(self):
        return self.m.body("dock").pos[:2].copy()

    # ---------- EVALUATION ONLY ----------
    def get_ground_truth(self):
        """EVALUATION ONLY: true (x, y, yaw), never used by the estimator or controller."""
        a = self._base_qadr
        x, y = self.d.qpos[a], self.d.qpos[a + 1]
        w, qx, qy, qz = self.d.qpos[a + 3:a + 7]
        yaw = math.atan2(2 * (w * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        return np.array([x, y, yaw])


    def marker_frame_in_world(self):
        """EVALUATION / DISPLAY ONLY: marker face position and direction in the world."""
        g = self.m.geom("dock_marker").id
        rot = self.d.geom_xmat[g].reshape(3, 3)
        normal = -rot[:2, 0]                                  # the face that looks into the room
        normal = normal / np.linalg.norm(normal)
        centre = self.d.geom_xpos[g][:2] + self.m.geom_size[g][0] * normal
        return centre, math.atan2(normal[1], normal[0])

    def get_ground_truth_marker_frame(self):
        """EVALUATION ONLY: true pose in the marker frame, to compare with the EKF."""
        (cx, cy), psi = self.marker_frame_in_world()
        x, y, yaw = self.get_ground_truth()
        dx, dy = x - cx, y - cy
        c, s = math.cos(psi), math.sin(psi)
        return np.array([c * dx + s * dy, -s * dx + c * dy,
                         (yaw - psi + math.pi) % (2 * math.pi) - math.pi])

    def marker_to_world(self, xy):
        """DISPLAY ONLY: marker frame -> world, for drawing."""
        (cx, cy), psi = self.marker_frame_in_world()
        c, s = math.cos(psi), math.sin(psi)
        return np.array([cx + c * xy[0] - s * xy[1], cy + s * xy[0] + c * xy[1]])
