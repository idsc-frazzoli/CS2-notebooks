function u = lqrController(x_hat, x_ref, u_eq, K)
% lqrController: Compute LQR control input for tracking.
%   x_hat: estimated state vector
%   x_ref: desired reference state vector [r_x; 0; r_y; 0; 0; 0]
%   u_eq: equilibrium control input (hover)
%   K: LQR gain matrix
%
% Returns:
%   u: control input vector [T; phi]

    % Compute state error (using the estimated state)
    e = x_hat - x_ref;
    % Control law: u = u_eq - K*e
    u = u_eq - K * e;
end
