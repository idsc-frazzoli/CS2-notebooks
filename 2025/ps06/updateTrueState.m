function x_new = updateTrueState(x, u, dt, m, I, g, l1)
% updateTrueState: Update the true state using the nonlinear dynamics.
%   x: current state [x; dx; y; dy; theta; dtheta]
%   u: control input [T; phi] (T in N, phi in rad)
%   dt: time step
%   m, I, g, l1: system parameters

T   = u(1);
phi = u(2);
theta = x(5);

% Nonlinear dynamics
x_ddot     = (T * sin(theta + phi)) / m;
y_ddot     = (- m*g + T * cos(theta + phi)) / m;
theta_ddot = (T * l1 * sin(phi)) / I;

% State derivative vector
x_dot = [ x(2);
          x_ddot;
          x(4);
          y_ddot;
          x(6);
          theta_ddot];

% Euler integration
x_new = x + dt * x_dot;
end
