import numpy as np
import matplotlib.pyplot as plt
from lqr_tracker import control, is_controllable

dt = 0.05
Q = np.diag([1/0.05**2, 1/0.05**2, 1/0.2**2])
R = np.diag([1/0.22**2, 1/2.84**2])

print("Controllable:", is_controllable(0.1, 0.2))

x = np.array([0.3, -0.2, 0.5])
traj, ref_traj = [], []
for k in range(600):
    t = k * dt
    vr, wr = 0.1, 0.2
    ref = np.array([0.5*np.sin(0.2*t), 0.5*(1-np.cos(0.2*t)), 0.2*t, vr, wr])
    u = control(x, ref, Q, R, dt)
    x = x + dt * np.array([u[0]*np.cos(x[2]), u[0]*np.sin(x[2]), u[1]])
    traj.append(x.copy()); ref_traj.append(ref[:3])

traj, ref_traj = np.array(traj), np.array(ref_traj)
plt.plot(ref_traj[:,0], ref_traj[:,1], '--', label='reference')
plt.plot(traj[:,0], traj[:,1], label='actual')
plt.axis('equal'); plt.legend(); plt.show()
