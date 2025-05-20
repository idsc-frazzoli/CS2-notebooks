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

# TODO: Add constants
m = 1.0 # kg
l = 0.25 # m 
I_z = 0.01 # kg*m^2
g = 9.81 # m/s^2

# TODO: Implement A and B matrices
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
# Process Noise
Q_lqe = np.diag([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
# Measurement Noise
R_lqe = np.diag([0.01, 0.01, 0.01])
# Cost on State
Q_lqr = np.diag([10, 1, 10, 1, 100, 10])
# Cost on Input
R_lqr = np.eye(2)

def discretize_euler_forward(A: np.ndarray, B: np.ndarray, C: np.ndarray, T: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    A_d = np.eye(A.shape[0]) + T * A
    B_d = T * B
    C_d = C
    return A_d, B_d, C_d

def discretize_euler_backward(A: np.ndarray, B: np.ndarray, C: np.ndarray, T: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    A_d = np.linalg.inv(np.eye(A.shape[0]) - T * A)
    B_d = T * A_d @ B
    C_d = C
    return A_d, B_d, C_d

def discretize_trapezoidal(A: np.ndarray, B: np.ndarray, C: np.ndarray, h: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    I = np.eye(A.shape[0])
    A_d = np.linalg.inv(I - 0.5 * h * A) @ (I + 0.5 * h * A)
    B_d = np.linalg.inv(I - 0.5 * h * A) @ (h * B)
    C_d = C.copy() if C is not None else None
    return A_d, B_d, C_d

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

dt = 0.05
# Choose discretization method
Ad, Bd, Cd = discretize_exact(A, B, C, dt)

@njit
def nonlinear_dynamics(t: float, state: np.ndarray, u: np.ndarray):
    x, x_dot, y, y_dot, theta, theta_dot = state
    T_1, T_2 = u + np.array([m*g/2, m*g/2])

    x_dd = (T_1 + T_2) / m * np.sin(theta)
    y_dd = (T_1 + T_2) / m * np.cos(theta) - g
    theta_dd = (T_1 - T_2) * l / I_z

    return np.array([x_dot, x_dd, y_dot, y_dd, theta_dot, theta_dd])

def lqe(Q_lqe: np.ndarray, R_lqe: np.ndarray) -> np.ndarray:
    P = solve_continuous_are(A.T, C.T, Q_lqe, R_lqe)
    L = P @ C.T @ np.linalg.inv(R_lqe)
    return L

def lqr(Q_lqr: np.ndarray, R_lqr: np.ndarray) -> np.ndarray:
    P = solve_continuous_are(A, B, Q_lqr, R_lqr)
    K = np.linalg.inv(R_lqr) @ B.T @ P
    return K

def dlqe(Q_lqe: np.ndarray, R_lqe: np.ndarray) -> np.ndarray:
    P = solve_discrete_are(Ad.T, Cd.T, Q_lqe, R_lqe)
    S = Cd @ P @ Cd.T + R_lqe
    Ld = P @ Cd.T @ np.linalg.inv(S)
    return Ld

def dlqr(Q_lqr: np.ndarray, R_lqr: np.ndarray) -> np.ndarray:
    P = solve_discrete_are(Ad, Bd, Q_lqr, R_lqr)
    Kd = np.linalg.inv(R_lqr + Bd.T @ P @ Bd) @ (Bd.T @ P @ Ad)
    return Kd

def linear_dynamics(t: float, state_hat: np.ndarray, u: np.ndarray, y: np.ndarray, L: np.ndarray):
    state_hat_dot = A @ state_hat + B @ u + L @ (y - C @ state_hat)
    x_dot, x_dd, y_dot, y_dd, theta_dot, theta_dd = state_hat_dot

    return np.array([x_dot, x_dd, y_dot, y_dd, theta_dot, theta_dd])

def discrete_linear_dynamics(t: float, state_hat: np.ndarray, u: np.ndarray, y: np.ndarray, Ld: np.ndarray):
    state_hat = Ad @ state_hat + Bd @ u + Ld @ (y - Cd @ state_hat)
    return state_hat

def sensor(t: float, state: np.ndarray) -> np.ndarray: 
    y = C @ state
    return y

@njit
def lqr_control(t: float, state_hat: np.ndarray, ref: np.ndarray, K: np.ndarray) -> np.ndarray:
    u = -K @ (state_hat - ref)
    return u

def dmpc_control(state_hat_k: np.ndarray, x_ref: np.ndarray, N: int, P_inf: np.ndarray) -> np.ndarray:
    nx = A.shape[0]
    nu = B.shape[1]
    x = cp.Variable((nx, N+1))
    u = cp.Variable((nu, N))
    u_ref = np.array([m*g/2, m*g/2])

    cost = 0
    constraints = [x[:, 0] == state_hat_k]

    for k in range(N):
        cost += cp.quad_form(x[:, k] - x_ref, Q_lqr) + cp.quad_form(u[:, k] - u_ref, R_lqr)
        constraints += [x[:, k+1] == Ad @ x[:, k] + Bd @ u[:, k]]
        constraints += [x[4, k] >= -0.4, x[4, k] <= 0.4]

    cost += cp.quad_form(x[:, N] - x_ref, P_inf)
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(solver=cp.OSQP, verbose=False)
    return u[:, 0].value.flatten()

def run_lqg(x_0: np.ndarray, x_ref: np.ndarray, t_final: float = 10.0, dt: float = 0.05) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    N = int(t_final / dt)
    
    # Set initial conditions
    state_0 = x_0.copy()
    state_hat_0 = x_0.copy()
    combined_state_0 = np.concatenate((state_0, state_hat_0))
    u_hist = np.zeros((2, N))
    
    L = lqe(Q_lqe, R_lqe)
    K = lqr(Q_lqr, R_lqr)

    # Precalculate noise
    process_noise = np.random.multivariate_normal(mean=np.zeros(6), cov=Q_lqe, size=N)
    sensor_noise = np.random.multivariate_normal(mean=np.zeros(3), cov=R_lqe, size=N)

    def combined_dynamics(t: float, combined_state: np.ndarray) -> np.ndarray:
        state = combined_state[:6]
        state_hat = combined_state[6:]
        idx = min(int(t / dt), N - 1)

        # Control law (state feedback on estimate
        u = lqr_control(t, state_hat, x_ref, K)
        u_hist[:, idx] = u

        # Evolve true nonlinear dynamics
        d_state = nonlinear_dynamics(t, state, u) + process_noise[idx]
        # Evolve linear dynamics with Kalman filter
        y = sensor(t, state) + sensor_noise[idx]
        d_state_hat = linear_dynamics(t, state_hat, u, y, L)

        return np.concatenate((d_state, d_state_hat))
    
    sol = solve_ivp(combined_dynamics, [0, t_final], combined_state_0, method='RK23', max_step=0.05)

    t = sol.t
    state_vector = sol.y[:6, :]
    state_hat_vector = sol.y[6:, :]

    return t, state_vector, state_hat_vector, u_hist

def run_dlqg(x_0: np.ndarray, x_ref: np.ndarray, t_final: float = 10.0, dt: float = 0.05)  -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    Ad, Bd, Cd = discretize_exact(A, B, C, dt)
    N = int(np.ceil(t_final / dt))
    t = np.linspace(0, N * dt, N + 1)

    # Set initial conditions
    state = x_0.copy()
    state_hat = x_0.copy()

    # Storage
    sol = np.zeros((12, N + 1))
    sol[:, 0] = np.concatenate((state, state_hat))
    u_hist = np.zeros((2, N))
    
    Ld = dlqe(Q_lqe, R_lqe)
    Kd = dlqr(Q_lqr, R_lqr)

    # Precalculate noise
    process_noise = np.random.multivariate_normal(mean=np.zeros(6), cov=Q_lqe, size=N)
    sensor_noise = np.random.multivariate_normal(mean=np.zeros(3), cov=R_lqe, size=N)

    for k in range(N):
        tk = t[k]
        u = lqr_control(tk, state_hat, x_ref, Kd)
        u_hist[:, k] = u
        
        # Nonlinear dynamics via RK23 propagate
        rk_sol = solve_ivp(lambda tau, s: nonlinear_dynamics(tk, s, u), [tk, tk + dt], state, method='RK23', max_step=dt, t_eval=[tk + dt])
        state = rk_sol.y[:, -1] + 0.2*process_noise[k]

        y = sensor(tk, state) + 0.2*sensor_noise[k]
        state_hat = discrete_linear_dynamics(tk, state_hat, u, y, Ld)

        sol[:, k+1] = np.concatenate((state, state_hat))

    state_vector = sol[:6, :]
    state_hat_vector = sol[6:, :]
    return t, state_vector, state_hat_vector, u_hist

def run_mpc(x_0: np.ndarray, x_ref: np.ndarray, N_prediction: int, t_final: float = 10.0, dt: float = 0.1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    Ad, Bd, Cd = discretize_exact(A, B, C, dt)

    # Time vector
    N = int(np.ceil(t_final / dt))
    t = np.linspace(0, N * dt, N + 1)

    # Set initial conditions
    state = x_0.copy()
    state_hat = x_0.copy()

    # Storage
    sol = np.zeros((12, N + 1))
    sol[:, 0] = np.concatenate((state, state_hat))
    u_hist = np.zeros((2, N))
    
    Ld = dlqe(Q_lqe, R_lqe)
    Kd = dlqr(Q_lqr, R_lqr)
    P_inf = solve_discrete_are(Ad, Bd, Q_lqr, R_lqr)

    # Precalculate noise
    process_noise = np.random.multivariate_normal(mean=np.zeros(6), cov=Q_lqe, size=N)
    sensor_noise = np.random.multivariate_normal(mean=np.zeros(3), cov=R_lqe, size=N)

    for k in range(N):
        tk = t[k]
        u = dmpc_control(state_hat, x_ref, N_prediction, P_inf)
        u_hist[:, k] = u
        
        # Nonlinear dynamics via RK23 propagate
        rk_sol = solve_ivp(lambda tau, s: nonlinear_dynamics(tk, s, u), [tk, tk + dt], state, method='RK23', max_step=dt, t_eval=[tk + dt])
        state = rk_sol.y[:, -1] + 0.1*process_noise[k]

        y = sensor(tk, state) + 0.1*sensor_noise[k]
        state_hat = linear_dynamics(tk, state_hat, u, y, Ld)

        sol[:, k+1] = np.concatenate((state, state_hat))


    state_vector = sol[:6, :]
    state_hat_vector = sol[6:, :]

    return t, state_vector, state_hat_vector, u_hist

x_0 = np.array([0.0, 0, 0, 0, 0, 0])
x_ref = np.array([2.0, 0, 2.0, 0, 0, 0])
t_final = 10.0
dt = 0.05
N_prediction = 5

#lqg_t, lqg_state_vector, lqg_state_hat_vector, lqg_u = run_lqg(x_0, x_ref, t_final, dt)
print("Finished LQG Simulation!")
dlqg_t, dlqg_state_vector, dlqg_state_hat_vector, dlqg_u = run_dlqg(x_0, x_ref, t_final, dt)
print("Finished DLQG Simulation!")
#mpc_t, mpc_state_vector, mpc_state_hat_vector, mpc_u = run_mpc(x_0, x_ref, N_prediction, t_final, dt)
print("Finished MPC Simulation!")