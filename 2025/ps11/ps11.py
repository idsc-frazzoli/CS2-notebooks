import numpy as np
import matplotlib.pyplot as plt

I_y = 3100 # kg*m^2
x_cm = 4.18 # m
l = x_cm
C_D = 0.1 
C_L = 0.034
g = 9.81 # m/s^2
I_sp0 = 300 # s
m0 = 1250
m_empty = 600

dt = 0.1

def altitude_to_pressure(altitude: float) -> float:
    return 101325 * np.exp(-altitude / 8500)

def thrust_to_mass_dot(thrust_input: np.ndarray, pressure: float) -> float:
    return - np.linalg.norm(thrust_input) / (I_sp0 * g) - (pressure * np.pi * 0.5**2) / (I_sp0 * g)

def f_a_input(state: np.ndarray) -> np.ndarray:
    x, x_dot, z, z_dot, theta, theta_dot = state
    velocity = np.sqrt(x_dot**2 + z_dot**2)
    q = 0.5 * 1.225 * velocity**2

    gamma = np.arctan2(x_dot, z_dot)
    alpha = theta - gamma

    C_A = C_D * np.cos(alpha) - C_L * np.sin(alpha)
    C_N = C_L * np.cos(alpha) + C_D * np.sin(alpha)

    S = np.pi * 0.5**2 # m^2

    return np.array([-q * C_A * S, -q * C_N * S])

def tau_a_input(state: np.ndarray, f_a_input: np.ndarray) -> float:
    x, x_dot, z, z_dot, theta, theta_dot = state
    f_ax, f_az = f_a_input
    # Center of pressure
    x_cp = 7.11 # m
    d = 0.5 # m

    tau_a = f_az * (x_cp - x_cm)

    return tau_a

def eom(t: float, state: np.ndarray, m: float, thrust_input: np.ndarray, f_a_input: np.ndarray, tau_a: float) -> np.ndarray:
    x, x_dot, z, z_dot, theta, theta_dot = state
    f_ax, f_az = f_a_input
    T_cos_mu, T_sin_mu = thrust_input

    dxdt = x_dot
    dx_dotdt = -g + f_ax / m + (np.cos(theta) * T_cos_mu + np.sin(theta) * T_sin_mu) / m
    dzdt = z_dot
    dz_dotdt = f_az / m + (-np.sin(theta) * T_cos_mu + np.cos(theta) * T_sin_mu) / m
    dthetadt = theta_dot
    dqdt = tau_a / I_y + (l * T_sin_mu) / I_y

    return np.array([dxdt, dx_dotdt, dzdt, dz_dotdt, dthetadt, dqdt])

def u_in(t: float, state: np.ndarray, m: float, f_a_input: np.ndarray, tau_a: float, ref: np.ndarray, v_in: np.ndarray) -> np.ndarray:
    x, x_dot, z, z_dot, theta, theta_dot = state
    f_ax, f_az = f_a_input
    x_d, x_dot_d, x_dd_d, z_d, z_dot_d, z_dd_d, theta_d, theta_dot_d, theta_dd_d = ref

    Lambda = np.array([[np.cos(theta)/m, np.sin(theta)/m],[0, l/I_y]])
    b = np.array([[-g + f_ax/m - x_dd_d], [tau_a/I_y - theta_dd_d]])

    v_in = np.reshape(v_in, (2, 1))
    u_in = np.linalg.inv(Lambda) @ b + np.linalg.inv(Lambda) @ v_in

    return u_in.T

def v_in(t: float, state: np.ndarray, ref: np.ndarray, k_1_x: float = 2.5, k_2_x: float = 4.5, k_1_theta: float = 12, k_2_theta: float = 10) -> np.ndarray:
    x, x_dot, z, z_dot, theta, theta_dot = state
    x_d, x_dot_d, x_dd_d, z_d, z_dot_d, z_dd_d, theta_d, theta_dot_d, theta_dd_d = ref

    e_x = x - x_d
    e_x_dot = x_dot - x_dot_d
    e_theta = theta - theta_d
    e_theta_dot = theta_dot - theta_dot_d

    return np.array([[
        -(1 + k_1_x*k_2_x) * e_x - (k_1_x + k_2_x) * e_x_dot
    ], [
        -(1 + k_1_theta*k_2_theta) * e_theta - (k_1_theta + k_2_theta) * e_theta_dot
    ]])

def theta_ref_from_u_out(t: float, state: np.ndarray, ref: np.ndarray, mass: float, f_a_input: np.ndarray, tau_a: float, u_out: float) -> np.ndarray:
    x, x_dot, z, z_dot, theta, theta_dot = state
    f_ax, f_az = f_a_input
    x_d, x_dot_d, x_dd_d, z_d, z_dot_d, z_dd_d, theta_d, theta_dot_d, theta_dd_d = ref

    a = -g + f_ax/mass - x_dd_d
    b = (I_y * theta_dd_d - tau_a)/(mass * l)

    theta_d_new = np.arccos(b/(np.sqrt(a**2 + u_out**2))) + np.arctan2(a, -u_out)
    theta_dot_d_new = (theta_d_new - theta_d)/dt
    theta_dd_d_new = (theta_dot_d_new - theta_dot_d)/dt

    return np.array([
        theta_d_new, theta_dot_d_new, theta_dd_d_new
    ])

def u_out(t: float, state: np.ndarray, m: float, f_a_input: np.ndarray, ref: np.ndarray, v_out: float) -> float:
    x, x_dot, z, z_dot, theta, theta_dot = state
    f_ax, f_az = f_a_input
    x_d, x_dot_d, x_dd_d, z_d, z_dot_d, z_dd_d, theta_d, theta_dot_d, theta_dd_d = ref

    u_out = z_dd_d - f_az/m + v_out

    return u_out

def v_out(t: float, state: np.ndarray, ref: np.ndarray, k_z: float = 0.61, k_z_dot: float = 1.11, k_i: float = 0.13) -> float:
    x, x_dot, z, z_dot, theta, theta_dot = state
    x_d, x_dot_d, x_dd_d, z_d, z_dot_d, z_dd_d, theta_d, theta_dot_d, theta_dd_d = ref

    e_z = z - z_d
    e_z_dot = z_dot - z_dot_d

    v_out = -k_z * e_z - k_z_dot * e_z_dot

    return v_out

def x_dd_ref(t: float) -> float:
    if t < 0:
        return 0.0
    elif t <= 30.0:
        return 10.0
    elif t <= 60.0:
        return 2.5*np.cos(2*np.pi*t/30.0)
    else:
        return 10.0
    
x_d = 0.0
x_dot_d = 0.0
x_dd_d = 0.0
z_d = 0.0
z_dot_d = 0.0
z_dd_d = 0.0
theta_d = 0.0
theta_dot_d = 0.0
theta_dd_d = 0.0

mass = m0
state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) # [x, x_dot, z, z_dot, theta, theta_dot]
thrust_input = np.array([0.0, 0.0]) # [T_cos_mu, T_sin_mu]

t = 0.0

time_hist = []
state_hist = []
ref_hist = []

while mass > m_empty:
    time_hist.append(t)
    # Compute reference trajectory
    x_dd_d = x_dd_ref(t)
    x_dot_d += x_dd_d * dt
    x_d += x_dot_d * dt
    ref = np.array([
        x_d, x_dot_d, x_dd_d, z_d, z_dot_d, z_dd_d, theta_d, theta_dot_d, theta_dd_d
    ])
    state_hist.append(state.copy())
    ref_hist.append(ref.copy())

    pressure = altitude_to_pressure(state[2])
    mass_dot = thrust_to_mass_dot(thrust_input, pressure)
    mass += 0.1*mass_dot

    f_a = f_a_input(state)
    tau_a = tau_a_input(state, f_a)

    v_out_val = v_out(t, state, ref)
    u_out_val = u_out(t, state, mass, f_a, ref, v_out_val)


    theta_d_triplet = theta_ref_from_u_out(t, state, ref, mass, f_a, tau_a, u_out_val)
    ref[6:9] = theta_d_triplet

    v_in_val = v_in(t, state, ref)
    u_in_val = u_in(t, state, mass, f_a, tau_a, ref, v_in_val)

    state_dot = eom(t, state, mass, u_in_val[0], f_a, tau_a)
    state += state_dot * dt

    t += dt
print("Reached end of loop!")

time_hist = np.array(time_hist)
state_hist = np.array(state_hist)
ref_hist = np.array(ref_hist)

# Labels for plotting
state_labels = ['x', 'x_dot', 'z', 'z_dot', 'theta', 'theta_dot']
ref_labels   = ['x_d', 'x_dot_d', 'x_dd_d', 'z_d', 'z_dot_d', 'z_dd_d', 'theta_d', 'theta_dot_d', 'theta_dd_d']
# Define the correct mapping from state index to reference index
ref_indices = [0, 1, 3, 4, 6, 7]

for i, label in enumerate(state_labels):
    plt.figure()
    plt.plot(time_hist, state_hist[:, i], label=f'{label} (actual)')
    plt.plot(time_hist, ref_hist[:, ref_indices[i]], '--', label=f'{ref_labels[ref_indices[i]]} (ref)')
    plt.xlabel('Time [s]')
    plt.ylabel(label)
    plt.title(f'{label} vs. {ref_labels[ref_indices[i]]}')
    plt.legend()
    plt.grid(True)
    plt.show()
