import numpy as np
import matplotlib.pyplot as plt

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
dt_outer = 0.1    # s
dt_inner = dt_outer / 10.0
t_final = 300.0   # s

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
    alpha = gamma - theta

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

def eom(state, m, thrust_input, f_a, tau_a):
    x_dot, z_dot, theta = state[1], state[3], state[4]
    f_ax, f_az = f_a
    T_cos_mu, T_sin_mu = thrust_input
    dxdt       = state[1]
    dx_dotdt   = -g + f_ax / m + (np.cos(theta) * T_cos_mu + np.sin(theta) * T_sin_mu) / m
    dzdt       = state[3]
    dz_dotdt   = f_az / m + (-np.sin(theta) * T_cos_mu + np.cos(theta) * T_sin_mu) / m
    dthetadt   = state[5]
    dqdt       = tau_a / I_y + (l * T_sin_mu) / I_y
    return np.array([dxdt, dx_dotdt, dzdt, dz_dotdt, dthetadt, dqdt])

def v_in(state, ref, k_x1=2.5, k_x2=4.5, k_th1=12, k_th2=10):
    e_x      = state[0] - ref[0]
    e_x_dot  = state[1] - ref[1]
    e_th     = state[4] - ref[6]
    e_th_dot = state[5] - ref[7]
    v1 = -(1 + k_x1*k_x2)*e_x - (k_x1 + k_x2)*e_x_dot
    v2 = -(1 + k_th1*k_th2)*e_th - (k_th1 + k_th2)*e_th_dot
    return np.array([v1, v2])

def u_in(state, m, f_a, tau_a, ref, v_in):
    theta = state[4]
    Lambda = np.array([[np.cos(theta)/m, np.sin(theta)/m],
                       [0, l/I_y]])
    b_vec  = np.array([[-g + f_a[0]/m - ref[2]], [tau_a/I_y - ref[8]]])
    return (np.linalg.inv(Lambda) @ b_vec + np.linalg.inv(Lambda) @ v_in.reshape(2,1)).flatten()

def v_out(state, ref, k1=0.61, k2=1.11):
    e_z     = state[2] - ref[3]
    e_z_dot = state[3] - ref[4]
    return -k1 * e_z - k2 * e_z_dot

def u_out(m, f_a, ref, v_out_val):
    return ref[5] - f_a[1]/m + v_out_val

def theta_ref_from_u_out(state, mass, f_a, tau_a, ref, u_out_val):
    # ref: [x_d, x_dot_d, x_dd_d, 0,0,0, theta_d, theta_dot_d, theta_dd_d]
    a = -g + f_a[0]/mass - ref[2]
    b = (I_y * ref[8] - tau_a)/(mass * l)
    val = np.clip(b/np.sqrt(a**2 + u_out_val**2), -1.0, 1.0)
    th_d = np.arccos(val) + np.arctan2(a, -u_out_val)
    th_dot_d = (th_d - ref[6]) / dt_outer
    th_dd_d  = (th_dot_d - ref[7]) / dt_outer
    return np.array([th_d, th_dot_d, th_dd_d])

def x_dd_ref(t):
    if t <= 30.0:
        return 10.0
    elif t <= 60.0:
        return 2.5 * np.cos(2 * np.pi * t / 30.0)
    else:
        return 10.0
    
def z_dd_ref(t):
    # 2.5 * [cos(2π/40 t) + 1] between t=20 and 60 s, else zero
    if 20.0 <= t <= 60.0:
        return 2.5 * (np.cos(2 * np.pi * t / 40.0) + 1.0)
    else:
        return 0.0

# --- Simulation multi-rate ---
t = 0.0
state = np.zeros(6)
mass = m0
x_d, x_dot_d = 0.0, 0.0
z_d, z_dot_d = 0.0, 0.0

# History
t_hist = []
y_hist = []

ref = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

while t < t_final and mass > m_empty:
    # Outer loop compute
    x_dd_d = x_dd_ref(t)
    z_dd_d = z_dd_ref(t)

    x_d     += x_dot_d * dt_outer
    x_dot_d += x_dd_d   * dt_outer
    z_d     += z_dot_d * dt_outer
    z_dot_d += z_dd_d   * dt_outer

    ref[0:6] = [x_d, x_dot_d, x_dd_d, z_d, z_dot_d, z_dd_d]
    f_a = f_a_input(state)
    tau_a = tau_a_input(state, f_a)
    v_out_val = v_out(state, ref)
    u_out_val = u_out(mass, f_a, ref, v_out_val)
    th_ref = theta_ref_from_u_out(state, mass, f_a, tau_a, ref, u_out_val)
    ref[6:9] = th_ref

    # Inner loop sub-steps
    for _ in range(10):
        f_a = f_a_input(state)
        tau_a = tau_a_input(state, f_a)
        v_in_val = v_in(state, ref)
        u_in_val = u_in(state, mass, f_a, tau_a, ref, v_in_val)
        state_dot = eom(state, mass, u_in_val, f_a, tau_a)
        p = altitude_to_pressure(state[2])
        mass_dot = thrust_to_mass_dot(np.zeros(2), p)

        state += state_dot * dt_inner
        mass  += mass_dot * dt_inner
        t     += dt_inner

        t_hist.append(t)
        y_hist.append(np.concatenate([state, [mass, x_d, x_dot_d]]))


# Convert history
hist = np.array(y_hist)
t_arr = np.array(t_hist)
X = hist[:, :6]
mass_arr = hist[:, 6]

# Reference values
ref_vals = {
    'x': hist[:, 7],
    'x_dot': hist[:, 8],
    'z': np.zeros_like(t_arr),
    'z_dot': np.zeros_like(t_arr),
    'theta': np.zeros_like(t_arr),
    'theta_dot': np.zeros_like(t_arr)
}

# Plot
state_labels = ['x', 'x_dot', 'z', 'z_dot', 'theta', 'theta_dot']
ref_labels   = ['x_d', 'x_dot_d', 'z_d', 'z_dot_d', 'theta_d', 'theta_dot_d']

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

