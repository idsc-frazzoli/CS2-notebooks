function x_hat_new = updateObserver(x_hat, u, y_meas, dt, A, B, C, L)
% updateObserver: Update the state estimate using a Luenberger observer.
%   x_hat: current estimated state
%   u: control input vector
%   y_meas: measured output vector
%   dt: time step
%   A, B, C: linearized system matrices
%   L: observer gain matrix computed via LQE
g    = 1.62;      % m/s^2
x_hat_dot = A*x_hat + B*u + L*(y_meas - C*x_hat) + [0; 0; 0; g; 0; 0];
x_hat_new = x_hat + dt * x_hat_dot;
end
