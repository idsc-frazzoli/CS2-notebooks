import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# --- constants & params ---
I_y = 3100.0      # kg·m²
x_cm = 4.18       # m
l = x_cm
C_D = 0.1 
C_L = 0.034
g = 9.81          # m/s²
I_sp0 = 300.0     # s
m0 = 1250.0       # kg
m_empty = 600.0   # kg
dt = 0.1          # s

def altitude_to_pressure(alt):
    return 101325 * np.exp(-alt/8500)

def thrust_to_mass_dot(T, p):
    return - np.linalg.norm(T)/(I_sp0*g) - (p*np.pi*0.5**2)/(I_sp0*g)

def f_a_input(state):
    x, x_dot, z, z_dot, theta, theta_dot = state
    velocity = np.sqrt(x_dot**2 + z_dot**2)
    if velocity < 1e-6:
        return np.array([0.0, 0.0])

    q_dyn = 0.5 * 1.225 * velocity**2
    gamma = np.arctan2(z_dot, x_dot)
    alpha = theta - gamma

    C_A = C_D * np.cos(alpha) - C_L * np.sin(alpha)
    C_N = C_L * np.cos(alpha) + C_D * np.sin(alpha)

    S = np.pi * 0.5**2  # reference area
    F_A = -q_dyn * C_A * S
    F_N = -q_dyn * C_N * S

    # Transform to inertial frame
    f_ax = F_A * np.cos(theta) - F_N * np.sin(theta)
    f_az = F_A * np.sin(theta) + F_N * np.cos(theta)

    return np.array([f_ax, f_az])

def tau_a_input(state, f_a):
    x, x_dot, z, z_dot, theta, theta_dot = state
    # rotate force into body frame
    f_ax, f_az = f_a
    f_ax_body = f_ax * np.cos(theta) + f_az * np.sin(theta)
    f_az_body = -f_ax * np.sin(theta) + f_az * np.cos(theta)
    x_cp = 7.11
    return f_az_body * (x_cp - x_cm)

def eom(t, state, m, thrust_input, f_a, tau_a):
    x, x_dot, z, z_dot, theta, theta_dot = state
    f_ax, f_az = f_a
    T_cos_mu, T_sin_mu = thrust_input
    dxdt = x_dot
    dx_dotdt = -g + f_ax / m + (np.cos(theta) * T_cos_mu + np.sin(theta) * T_sin_mu) / m
    dzdt = z_dot
    dz_dotdt = f_az / m + (-np.sin(theta) * T_cos_mu + np.cos(theta) * T_sin_mu) / m
    dthetadt = theta_dot
    dqdt = tau_a / I_y + (l * T_sin_mu) / I_y
    return np.array([dxdt, dx_dotdt, dzdt, dz_dotdt, dthetadt, dqdt])

def u_in(t, state, m, f_a, tau_a, ref, v_in):
    x, x_dot, z, z_dot, theta, theta_dot = state
    x_d, x_dot_d, x_dd_d, z_d, z_dot_d, z_dd_d, theta_d, theta_dot_d, theta_dd_d = ref
    Lambda = np.array([[np.cos(theta)/m, np.sin(theta)/m],
                       [0, l/I_y]])
    b = np.array([[-g + f_a[0]/m - x_dd_d], [tau_a/I_y - theta_dd_d]])
    v_in = np.reshape(v_in, (2, 1))
    u = np.linalg.inv(Lambda) @ b + np.linalg.inv(Lambda) @ v_in
    return u.T

def v_in(t, state, ref, k_1_x=2.5, k_2_x=4.5, k_1_theta=12, k_2_theta=10):
    x, x_dot, z, z_dot, theta, theta_dot = state
    x_d, x_dot_d, x_dd_d, z_d, z_dot_d, z_dd_d, theta_d, theta_dot_d, theta_dd_d = ref
    e_x = x - x_d
    e_x_dot = x_dot - x_dot_d
    e_theta = theta - theta_d
    e_theta_dot = theta_dot - theta_dot_d
    return np.array([[ -(1 + k_1_x*k_2_x)*e_x - (k_1_x + k_2_x)*e_x_dot ],
                     [ -(1 + k_1_theta*k_2_theta)*e_theta - (k_1_theta + k_2_theta)*e_theta_dot ]])

def theta_ref_from_u_out(t, state, ref, mass, f_a, tau_a, u_out):
    x, x_dot, z, z_dot, theta, theta_dot = state
    x_d, x_dot_d, x_dd_d, z_d, z_dot_d, z_dd_d, theta_d, theta_dot_d, theta_dd_d = ref
    a = -g + f_a[0]/mass - x_dd_d
    b = (I_y * theta_dd_d - tau_a)/(mass * l)
    val = np.clip(b/np.sqrt(a**2 + u_out**2), -1.0, 1.0)
    theta_d_new = np.arccos(val) + np.arctan2(a, -u_out)
    theta_dot_d_new = (theta_d_new - theta_d)/dt
    theta_dd_d_new = (theta_dot_d_new - theta_dot_d)/dt
    return np.array([theta_d_new, theta_dot_d_new, theta_dd_d_new])

def u_out(t, state, m, f_a, ref, v_out):
    z_dd_d = ref[5]
    return z_dd_d - f_a[1]/m + v_out

def v_out(t, state, ref, k_z=0.61, k_z_dot=1.11):
    z, z_dot = state[2], state[3]
    z_d, z_dot_d = ref[3], ref[4]
    e_z = z - z_d
    e_z_dot = z_dot - z_dot_d
    return -k_z * e_z - k_z_dot * e_z_dot

def x_dd_ref(t):
    if t <= 30.0:
        return 10.0
    elif t <= 60.0:
        return 2.5*np.cos(2*np.pi*t/30.0)
    else:
        return 10.0

# initial conditions
y0 = np.zeros(9)
y0[6] = m0  # mass
# y0[7], y0[8] are x_d, x_dot_d = 0

# ODE system
def full_deriv(t, y):
    state = y[:6]
    mass = y[6]
    x_d, x_dot_d = y[7], y[8]
    # ref update
    x_dd_d = x_dd_ref(t)
    dx_d_dt = x_dot_d
    dx_dot_d_dt = x_dd_d
    ref = np.array([x_d, x_dot_d, x_dd_d, 0, 0, 0, 0, 0, 0])
    # aero & mass flow
    p = altitude_to_pressure(state[2])
    mass_dot = thrust_to_mass_dot(np.zeros(2), p) * dt
    f_a = f_a_input(state)
    tau_a = tau_a_input(state, f_a)
    # control
    v_out_val = v_out(t, state, ref)
    u_out_val = u_out(t, state, mass, f_a, ref, v_out_val)
    th_ref = theta_ref_from_u_out(t, state, ref, mass, f_a, tau_a, u_out_val)
    ref[6:9] = th_ref
    v_in_val = v_in(t, state, ref)
    u_in_val = u_in(t, state, mass, f_a, tau_a, ref, v_in_val)
    # dynamics
    state_dot = eom(t, state, mass, u_in_val[0], f_a, tau_a)
    return np.concatenate([state_dot, [mass_dot, dx_d_dt, dx_dot_d_dt]])

# integrate
sol = solve_ivp(full_deriv, (0, 300), y0, method='RK23', max_step=dt,
                events=lambda t, y: y[6] - m_empty)

t_arr = sol.t
X = sol.y[:6].T
mass_arr = sol.y[6]

# Define labels
state_labels = ['x', 'x_dot', 'z', 'z_dot', 'theta', 'theta_dot']
ref_labels = ['x_d', 'x_dot_d', 'z_d', 'z_dot_d', 'theta_d', 'theta_dot_d']

# Build reference arrays
ref_vals = {
    'x': sol.y[7],
    'x_dot': sol.y[8],
    'z': np.zeros_like(t_arr),
    'z_dot': np.zeros_like(t_arr),
    'theta': np.zeros_like(t_arr),
    'theta_dot': np.zeros_like(t_arr)
}

# Plot each state vs. its reference
for idx, label in enumerate(state_labels):
    plt.figure()
    plt.plot(t_arr, X[:, idx], label=f'{label} (actual)')
    plt.plot(t_arr, ref_vals[label], '--', label=f'{ref_labels[idx]} (ref)')
    plt.xlabel('Time [s]')
    plt.ylabel(label)
    plt.title(f'{label} vs. {ref_labels[idx]}')
    plt.legend()
    plt.grid(True)

plt.show()