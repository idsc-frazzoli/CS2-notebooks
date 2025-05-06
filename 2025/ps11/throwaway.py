import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# --- vehicle & environment parameters ---
I_y      = 3100.0     # kg·m²
l        = 4.18       # m  (distance thrust line to CG)
g        = 9.81       # m/s²
I_sp0    = 300.0      # s
g0       = 9.81       # m/s²
m0       = 1250.0     # kg  (initial mass)
m_empty  = 600.0      # kg  (dry mass)
C_D      = 0.1
C_L      = 0.034

A_ref    = 1.0       # m² (reference area for aero)
A_e      = 0.1       # m² (nozzle exit area)

def altitude_to_pressure(alt):
    return 101325.0 * np.exp(-alt/8500.0)

def aero_forces(vx, vz, alt):
    rho = altitude_to_pressure(alt) / (287.0 * 288.0)
    v = np.hypot(vx, vz)
    if v < 1e-3:
        return 0.0, 0.0
    F_D = 0.5 * rho * v**2 * C_D * A_ref
    F_L = 0.5 * rho * v**2 * C_L * A_ref
    fx = -F_D * (vx/v) + F_L * (-vz/v)
    fz = -F_D * (vz/v) + F_L * ( vx/v)
    return fx, fz

def aero_moment(q, alt):
    return 0.0

# reference accelerations
def xddot_ref(t):
    if 0 <= t <= 30.0:
        return 10.0
    elif 30.0 < t <= 60.0:
        return 2.5 * np.cos(2*np.pi/30.0 * t) + 7.5
    else:
        return 10.0

def zddot_ref(t):
    return 0.0

# gains
k1_x, k2_x = 2.5, 4.5
k1_th, k2_th = 12.0, 10.0
k_z, k_dz, k_iz = 0.61, 1.11, 0.13

# integrator dynamics
def dynamics(t, state):
    x, xdot, z, zdot, theta, q, m, ez_int = state
    f_ax, f_az = aero_forces(xdot, zdot, x)
    tau_a      = aero_moment(q, x)

    # reference and errors for x
    xdd_d = xddot_ref(t)
    # errors wrt zero integrals (we only compare accelerations)
    ex = x
    edot_x = xdot

    # reference and errors for z
    ez = z
    edot_z = zdot

    # outer loop
    v_out = -k_z*ez - k_dz*edot_z - k_iz*ez_int
    u_out = - f_az/m + v_out

    # theta reference
    a = -g + f_ax/m - xdd_d
    b = 0.0
    theta_d = np.arccos(b/np.hypot(a, u_out)) + np.arctan2(a, -u_out)

    # inner loop
    b_in = np.array([-g + f_ax/m - xdd_d,
                     tau_a/I_y])
    Lam  = np.array([[np.cos(theta)/m, np.sin(theta)/m],
                     [0.0,               l/I_y       ]])
    v_in = np.array([-(1+k1_x*k2_x)*ex - (k1_x+k2_x)*edot_x,
                     -(1+k1_th*k2_th)*(theta-theta_d) - (k1_th+k2_th)*q])
    u_in = np.linalg.solve(Lam, -b_in + v_in)
    T_cos, T_sin = u_in
    T   = np.hypot(T_cos, T_sin)

    # state derivatives
    xdd = -g + f_ax/m + (T_cos*np.cos(theta) + T_sin*np.sin(theta))
    zdd =        f_az/m + (-T_cos*np.sin(theta) + T_sin*np.cos(theta))
    thdot = q
    qdot  = tau_a/I_y + (l/I_y)*T_sin
    p_a = altitude_to_pressure(x)
    m_dot = -T/(I_sp0*g0) - (p_a*A_e)/(I_sp0*g0)

    return [xdot, xdd, zdot, zdd, thdot, qdot, m_dot, edot_z]

# simulate
t_final = 100.0
state0 = [0,0, 0,0, 0,0, m0, 0.0]
sol = solve_ivp(dynamics, [0, t_final], state0, atol=1e-6, rtol=1e-8, max_step=0.05)

# time and states
t = sol.t
x, xdot, z, zdot, theta = sol.y[0], sol.y[1], sol.y[2], sol.y[3], sol.y[4]
ez_int = sol.y[7]

# compute reference trajectories by integration
xdd_ref_vec = np.array([xddot_ref(ti) for ti in t])
xdot_ref = np.concatenate([[0], np.cumtrapz(xdd_ref_vec, t)])
x_ref    = np.concatenate([[0], np.cumtrapz(xdot_ref, t)])
z_ref    = np.zeros_like(t)
zdot_ref = np.zeros_like(t)

# recompute theta_ref from simulation data
theta_ref = np.zeros_like(t)
for i, ti in enumerate(t):
    ax, az = aero_forces(xdot[i], zdot[i], x[i])
    m_i = sol.y[6,i]
    a = -g + ax/m_i - xdd_ref_vec[i]
    ez_i = z[i]
    edz_i = zdot[i]
    ezint_i = ez_int[i]
    v_out = -k_z*ez_i - k_dz*edz_i - k_iz*ezint_i
    u_out = - az/m_i + v_out
    theta_ref[i] = np.arccos(0.0/np.hypot(a, u_out)) + np.arctan2(a, -u_out)

# plot states vs references
plt.figure(figsize=(10,8))

plt.subplot(3,1,1)
plt.plot(t, x, label='x (state)')
plt.plot(t, x_ref, '--', label='x_ref')
plt.ylabel('Altitude [m]')
plt.legend()

plt.subplot(3,1,2)
plt.plot(t, z, label='z (state)')
plt.plot(t, z_ref, '--', label='z_ref')
plt.ylabel('Downrange [m]')
plt.legend()

plt.subplot(3,1,3)
plt.plot(t, np.rad2deg(theta), label='θ (deg)')
plt.plot(t, np.rad2deg(theta_ref), '--', label='θ_ref (deg)')
plt.ylabel('Pitch [deg]')
plt.xlabel('Time [s]')
plt.legend()

plt.tight_layout()
plt.show()
