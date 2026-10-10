import numpy as np
from scipy.linalg import solve_discrete_are
from scipy.signal import cont2discrete

def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi

def error_model(v_r, w_r):
    A = np.array([[0,    w_r, 0],
                  [-w_r, 0,   v_r],
                  [0,    0,   0]])
    B = np.array([[-1, 0],
                  [ 0, 0],
                  [ 0, -1]])
    return A, B

def is_controllable(v_r, w_r):
    A, B = error_model(v_r, w_r)
    C = np.hstack([B, A @ B, A @ A @ B])
    return np.linalg.matrix_rank(C) == 3

def lqr_gain(v_r, w_r, dt, Q, R):
    A, B = error_model(v_r, w_r)
    Ad, Bd, *_ = cont2discrete((A, B, np.eye(3), np.zeros((3, 2))), dt)
    P = solve_discrete_are(Ad, Bd, Q, R)
    return np.linalg.solve(R + Bd.T @ P @ Bd, Bd.T @ P @ Ad)

def control(x_hat, ref, Q, R, dt, v_max=0.22, w_max=2.84):
    """x_hat: ESTIMATED [x, y, theta]. ref: [xr, yr, thr, vr, wr]."""
    x, y, th = x_hat
    xr, yr, thr, vr, wr = ref
    dx, dy = xr - x, yr - y
    e = np.array([ np.cos(th) * dx + np.sin(th) * dy,
                  -np.sin(th) * dx + np.cos(th) * dy,
                   wrap(thr - th)])
    K = lqr_gain(vr, wr, dt, Q, R)
    u = np.array([vr, wr]) - K @ e
    return np.clip(u, [-v_max, -w_max], [v_max, w_max])
