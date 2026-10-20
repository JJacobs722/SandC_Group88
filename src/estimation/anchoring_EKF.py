#project estimattion goes here :D


#base layout of code

#combining the wheel turns with whats seen by the camera (the aruco marker)
#Setup EKF fusing wheel-encoder odometry
    #includes range and bearing based on visual placement of aruco marker (x,y,theta)

#wheel odometry tells you how far the robot has travelled through the encoder :P
#Begin with an estimation (guess) then be like okay well how can I recover from this which is what
#the EKF is for 

#NO GROUND TRUTH EVER

#imports taken from assessment 2 file!
from __future__ import annotations  # so 'float | None' works on older python
import numpy as np
from dataclasses import dataclass, field



def angle_wrap(a): #wrapping the angle to include -pi and exclude pi
    return (a + np.pi) % (2 * np.pi) - np.pi

@dataclass
class EKF_build:
    #platform geometry taken from model -> Needs update based on actual radius and base
    wheel_radius: float = 0.033
    wheel_base: float = 0.160
    #processing dat noise fr fr
    k_wheel: float = 1e-3 #in m, variance per m travelled per wheel
    min_wheel_var: float = 1e-9 #keeps Q non-zero when stationary
    k_turn: float = 0.0 #m^2 per rad turned: the waffle shuffles sideways when it turns, the wheels can't see this
    q_floor: np.ndarray = field(default_factory = lambda: np.diag([1e-8, 1e-8, 1e-8]))
    #measurement noise
    R_range_bearing: np.ndarray = field(default_factory = lambda: np.diag([0.05**2, np.deg2rad(3)**2]))
    R_pose: np.ndarray = field(default_factory = lambda: np.diag([0.05**2, 0.05**2, np.deg2rad(5)**2]))
    #starting uncertainty
    P0: np.ndarray = field(default_factory = lambda: np.diag([0.05**2, 0.05**2, np.deg2rad(5)**2]))
    #innovation gate: chi-square of 99% confidence
    #used for NIS comparison against threshold with 2 DOF = 9.21 and 3 DOF = 11.34
    gate_range_bearing: float | None = 9.21
    gate_pose: float | None = 11.34

class Odometry_Wheels:
    #turns wheel angles from encoders into per-step increments
    def __init__(self, phi_l0: float, phi_r0: float):
        self._prev = (phi_l0, phi_r0)
    def increments(self, phi_l: float, phi_r: float): #left and right wheel increments
        d1 = phi_l - self._prev[0]
        dr = phi_r - self._prev[1]
        self._prev = (phi_l, phi_r)
        return d1, dr

class EKF:
    def __init__(self, x0, cfg: EKF_build | None = None, P0 = None):
        self.cfg = cfg or EKF_build()
        self.x = np.array(x0, dtype = float).reshape(3)
        self.x[2] = angle_wrap(self.x[2])
        self.P = np.array(P0 if P0 is not None else self.cfg.P0, dtype = float)

        #diagnostics shit for reporting if needed
        self.last_nis = None
        self.last_innovation = None
        self.last_update_accepted = None

    #predicting stage aight
    def predict(self, dphi_l: float, dphi_r: float):
        #propogating with wheel angle increments since the last call
        c = self.cfg #calling to EKF_build -> object call where cfg's are configure :P
        r, L = c.wheel_radius, c.wheel_base
        ds_l, ds_r = r*dphi_l, r*dphi_r
        ds = 0.5*(ds_r + ds_l)
        dth = (ds_r - ds_l) / L

        th = self.x[2]
        thm = th + 0.5*dth  #midpoint heading -> more accurate than old heading :D
        cm, sm = np.cos(thm), np.sin(thm)

        #stage propagation
        self.x = self.x + np.array([ds * cm, ds * sm, dth])
        self.x[2] = angle_wrap(self.x[2])

        #jacobian state
        F = np.array([[1, 0, -ds*sm], [0, 1, ds*cm], [0, 0, 1]])

        #jacobian wheel arc lengths which is ds_l and ds_r
        G = np.array([[0.5*cm + ds / (2*L) * sm, 0.5*cm - ds / (2*L) * sm],
                      [0.5 * sm - ds / (2*L) * cm, 0.5 * sm + ds / (2*L) * cm],
                      [-1 / L, 1 / L]])
        Sigma_u = np.diag([c.k_wheel * abs(ds_l) + c.min_wheel_var, c.k_wheel * abs(ds_r) + c.min_wheel_var])
        self.P = F @ self.P @ F.T + G @ Sigma_u @ G.T + c.q_floor
        #turning slip: the body sits behind the wheels and drags, so turning moves the robot a bit
        #in a direction the wheel model doesn't predict. grow the position uncertainty with |turn|
        self.P[:2, :2] += c.k_turn * abs(dth) * np.eye(2)
        self.P = 0.5*(self.P + self.P.T)

    def update_range_bearing(self, z, landmark_xy, R = None):

        #z = [range, bearing] of the marker measured in robot's frame with bearing going left from robot heading
        #landmark is the known marker position in world frame
        #returns true if hte update was successfully applied

        R = self.cfg.R_range_bearing if R is None else R
        mx, my = landmark_xy
        dx, dy = mx - self.x[0], my - self.x[1]
        q = dx*dx + dy*dy
        rho = np.sqrt(q)

        if rho < 1e-6:
            return False
        zhat = np.array([rho, angle_wrap(np.arctan2(dy, dx) - self.x[2])])
        H = np.array([[-dx / rho, -dy / rho, 0], [dy / q, -dx / q, -1]])
        return self._update(np.asarray(z, float), zhat, H, R, angle_idx=(1,), gate=self.cfg.gate_range_bearing)

    def update_pose(self, z, R=None):
        #z = [x,y,theta] robot pose in world frame
        R = self.cfg.R_pose if R is None else R
        return self._update(np.asarray(z, float), self.x.copy(), np.eye(3), R, angle_idx=(2,), gate=self.cfg.gate_pose)

    def update_marker(self, z, R):
        #z = [range, bearing, facing] of the marker seen from the robot
        #the world frame IS the marker frame: marker at (0,0), its face pointing along +x
        #so no map is needed, the marker defines where (0,0) is
        x, y, th = self.x
        q = x*x + y*y
        rho = np.sqrt(q)
        if rho < 1e-6:
            return False
        zhat = np.array([rho,
                         angle_wrap(np.arctan2(-y, -x) - th),   #direction to the marker
                         angle_wrap(-th)])                      #marker face points along +x
        H = np.array([[x / rho, y / rho, 0],
                      [-y / q, x / q, -1],
                      [0, 0, -1]])
        return self._update(np.asarray(z, float), zhat, H, R, angle_idx=(1, 2), gate=self.cfg.gate_pose)

    def _update(self, z, zhat, H, R, angle_idx=(), gate=None):
        y = z - zhat
        for i in angle_idx:
            y[i] = angle_wrap(y[i])
        S = H @ self.P @ H.T + R
        nis = float(y @ np.linalg.solve(S, y))
        self.last_innovation, self.last_nis = y, nis

        if gate is not None and nis > gate:
            self.last_update_accepted = False
            return False
        K = np.linalg.solve(S, H @ self.P).T
        self.x = self.x + K @ y
        self.x[2] = angle_wrap(self.x[2])
        I_KH = np.eye(3) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        self.last_update_accepted = True
        return True

        #one-call API
    def step(self, phi_l: float, phi_r: float, detection=None, landmark_xy=None, R=None):
        #one place to call once per control tick where:
            #phi_l and r -> absolute wheel angles in rad
            #detection -> none if the marker is not visible, (range, bearing) with landmark_xy,
            #             or (range, bearing, facing) in the marker frame
            #landmark_xy -> known world position of marker
            #R -> optional measurement covariance for this detection, if none then EKF_build defaults used
        if not hasattr(self, "_odom"):
            self._odom = Odometry_Wheels(phi_l, phi_r) #object call
        else:
            self.predict(*self._odom.increments(phi_l, phi_r))

        if detection is not None:
            detection = np.asarray(detection, dtype = float).ravel()
            if detection.size == 2:
                if landmark_xy is None:
                    raise ValueError("range/bearing detection needs landmark_xy value")
                self.update_range_bearing(detection, landmark_xy, R)
            elif detection.size == 3:
                #[range, bearing, facing] of the marker, marker frame
                self.update_marker(detection, R)
            else:
                raise ValueError("detection must have 2 (range and bearing) or 3 (pose) entries :P")
        return self.state, self.covariance

    #interface stuff
    @property
    def state(self):
        #this is what the controller consumes mmm yummy
        return self.x.copy()

    @property
    def covariance(self):
        return self.P.copy()

def pose_from_marker(z):
    #one sighting [range, bearing, facing] -> robot pose (x, y, theta) in the marker frame
    r, b, f = z
    th = angle_wrap(-f)                         #marker face is +x, so robot heading = -facing
    px, py = r * np.cos(b), r * np.sin(b)       #marker position seen from the robot
    c, s = np.cos(th), np.sin(th)
    return np.array([-(c * px - s * py), -(s * px + c * py), th])


def start_from_marker(z, R, cfg=None):
    #start the EKF from the first good sighting, no guess of the start needed
    #P0 comes from pushing the measurement noise R through pose_from_marker (numerical jacobian)
    z = np.asarray(z, float)
    x0 = pose_from_marker(z)
    J = np.zeros((3, 3))
    for i in range(3):
        dz = np.zeros(3)
        dz[i] = 1e-6
        d = pose_from_marker(z + dz) - x0
        d[2] = angle_wrap(d[2])
        J[:, i] = d / 1e-6
    P0 = J @ np.asarray(R, float) @ J.T + np.diag([1e-4, 1e-4, 1e-4])
    return EKF(x0, cfg, P0=P0)


#cheeky self-test to see if anything dies, good for testing before merging to main
if __name__ == "__main__":
    # Pure-numpy sanity check: no MuJoCo needed. Run:  python ekf.py
    rng = np.random.default_rng(0)
    cfg = EKF_build()
    r, L, dt = cfg.wheel_radius, cfg.wheel_base, 0.02
    landmark = np.array([1.5, 0.0])
 
    truth = np.array([0.0, 0.0, 0.0])
    ekf = EKF(truth + [0.03, -0.03, 0.05])
    dead = EKF(truth + [0.03, -0.03, 0.05])        # predict-only, for comparison
    nis_log = []
    phi_l = phi_r = 0.0                            # simulated absolute encoder angles
    ekf.step(phi_l, phi_r)                         # first call just records the angles
 
    for k in range(1500):
        v, w = 0.12, 0.25
        dphi_r = (v + 0.5 * w * L) * dt / r
        dphi_l = (v - 0.5 * w * L) * dt / r
        # true motion
        ds, dth = v * dt, w * dt
        truth += [ds * np.cos(truth[2] + dth / 2), ds * np.sin(truth[2] + dth / 2), dth]
        # noisy / biased encoders
        nl = dphi_l * (1 + 0.02 * rng.standard_normal()) * 1.01
        nr = dphi_r * (1 + 0.02 * rng.standard_normal()) * 0.99
        dead.predict(nl, nr)
        phi_l += nl
        phi_r += nr
 
        z = None
        if k % 10 == 0:                            # camera at 5 Hz
            d = landmark - truth[:2]
            z = [np.hypot(*d) + 0.05 * rng.standard_normal(),
                 angle_wrap(np.arctan2(d[1], d[0]) - truth[2]) + np.deg2rad(3) * rng.standard_normal()]
        state, cov = ekf.step(phi_l, phi_r, z, landmark)
        if z is not None and ekf.last_update_accepted:
            nis_log.append(ekf.last_nis)
 
    print("EKF  position error [m]:", np.linalg.norm(ekf.x[:2] - truth[:2]))
    print("Dead-reckoning error [m]:", np.linalg.norm(dead.x[:2] - truth[:2]))
    print("Mean NIS (should be ~2):", np.mean(nis_log))