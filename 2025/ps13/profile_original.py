import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import ipywidgets as widgets
import control as ctrl
import cvxpy as cp
from scipy.linalg import expm, solve_continuous_are, solve_discrete_are
from scipy.integrate import solve_ivp
from IPython.display import display, clear_output
from typing import Tuple
from numba import njit

m = 1.0 # kg
l = 0.25 # m 
I_z = 0.01 # kg*m^2
g = 9.81 # m/s^2

def discretize_exact(A: np.ndarray, B: np.ndarray, C: np.ndarray, h: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = A.shape[0]
    m = B.shape[1]
    M = np.zeros((n + m, n + m))
    M[:n, :n] = A
    M[:n, n:] = B
    expM = expm(h * M)
    Ad = expM[:n, :n]
    Bd = expM[:n, n:]
    Cd = C.copy() if C is not None else None
    return Ad, Bd, Cd

A = np.array([[0,1,0,0,0,0],
            [0,0,0,0,g,0],
            [0,0,0,1,0,0],
            [0,0,0,0,0,0],
            [0,0,0,0,0,1],
            [0,0,0,0,0,0]])
B = np.array([[0,0],
            [0,0],
            [0,0],
            [1/m,1/m],
            [0,0],
            [l/I_z,-l/I_z]])
C = np.array([[1,0,0,0,0,0],
            [0,0,1,0,0,0],
            [0,0,0,0,1,0]])

# Choose timestep
dt = 0.1
# Choose discretization method
Ad, Bd, Cd = discretize_exact(A, B, C, dt)

# Process Noise
Q_lqe = np.diag([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
# Measurement Noise
R_lqe = np.diag([0.01, 0.01, 0.01])
# Cost on State
Q_lqr = np.diag([10, 1, 10, 1, 100, 10])
# Cost on Input
R_lqr = np.eye(2)

@njit
def nonlinear_dynamics(t: float, state: np.ndarray, u: np.ndarray):
    x, x_dot, y, y_dot, theta, theta_dot = state
    T_1, T_2 = u + np.array([m*g/2, m*g/2])

    x_dd = (T_1 + T_2) / m * np.sin(theta)
    y_dd = (T_1 + T_2) / m * np.cos(theta) - g
    theta_dd = (T_1 - T_2) * l / I_z

    return np.array([x_dot, x_dd, y_dot, y_dd, theta_dot, theta_dd])

def dlqe(Q_lqe: np.ndarray, R_lqe: np.ndarray) -> np.ndarray:
    P = solve_discrete_are(Ad.T, Cd.T, Q_lqe, R_lqe)
    S = Cd @ P @ Cd.T + R_lqe
    Ld = P @ Cd.T @ np.linalg.inv(S)
    return Ld

def dlqr(Q_lqr: np.ndarray, R_lqr: np.ndarray) -> np.ndarray:
    P = solve_discrete_are(Ad, Bd, Q_lqr, R_lqr)
    Kd = np.linalg.inv(R_lqr + Bd.T @ P @ Bd) @ (Bd.T @ P @ Ad)
    return Kd

def linear_dynamics(t: float, state_hat: np.ndarray, u: np.ndarray, y: np.ndarray, Ld: np.ndarray):
    state_hat = Ad @ state_hat + Bd @ u + Ld @ (y - Cd @ state_hat)
    return state_hat

def sensor(t: float, state: np.ndarray) -> np.ndarray: 
    y = Cd @ state
    return y

@njit
def lqr_control(t: float, state_hat: np.ndarray, ref: np.ndarray, K: np.ndarray) -> np.ndarray:
    u = -K @ (state_hat - ref)
    return u


x_0 = np.array([0.0, 0, 0, 0, 0, 0])
x_ref = np.array([2.0, 0, 2.0, 0, 0, 0])
t_final = 10.0
dt = 0.05
N_prediction = 5


# Discretize once
Ad, Bd, Cd = discretize_exact(A, B, C, dt)

# Time vector
N = int(np.ceil(t_final / dt))
t = np.linspace(0, N * dt, N + 1)

# Set initial conditions
state = x_0
state_hat = x_0

# Storage
sol = np.zeros((12, N + 1))
sol[:, 0] = np.concatenate((state, state_hat))

Ld = dlqe(Q_lqe, R_lqe)
Kd = dlqr(Q_lqr, R_lqr)

# Precalculate noise
process_noise = np.random.multivariate_normal(mean=np.zeros(6), cov=Q_lqe, size=N)
sensor_noise = np.random.multivariate_normal(mean=np.zeros(3), cov=R_lqe, size=N)

for k in range(N):
    tk = t[k]
    u = lqr_control(tk, state_hat, x_ref, Kd)
    
    # Nonlinear dynamics via RK23 propagate
    rk_sol = solve_ivp(lambda tau, s: nonlinear_dynamics(tk, s, u), [tk, tk + dt], state, method='RK23', max_step=dt, t_eval=[tk + dt])
    state = rk_sol.y[:, -1] + 0.2*process_noise[k]

    y = sensor(tk, state) + 0.2*sensor_noise[k]
    state_hat = linear_dynamics(tk, state_hat, u, y, Ld)

    sol[:, k+1] = np.concatenate((state, state_hat))


state_vector = sol[:6, :]
state_hat_vector = sol[6:, :]

# Plotting
fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
axes[0].plot(t, state_vector[0], label='x')
axes[0].plot(t, state_hat_vector[0], label='x_hat')
axes[0].axhline(x_ref[0], color='r', linestyle='--', label='x_ref')
axes[0].set_ylabel('X Position (m)')
axes[0].set_ylim(-5, 5)
axes[0].legend()

axes[1].plot(t, state_vector[2], label='y')
axes[1].plot(t, state_hat_vector[2], label='y_hat')
axes[1].axhline(x_ref[2], color='r', linestyle='--', label='y_ref')
axes[1].set_ylabel('Y Position (m)')
axes[1].set_ylim(-5, 5)
axes[1].legend()

axes[2].plot(t, state_vector[4], label='theta')
axes[2].plot(t, state_hat_vector[4], label='theta_hat')
axes[2].axhline(x_ref[4], color='r', linestyle='--', label='theta_ref')
axes[2].set_ylabel('Pitch θ (rad)')
axes[2].set_ylim(-0.5*np.pi, 0.5*np.pi)
axes[2].legend()

plt.tight_layout()
plt.show()