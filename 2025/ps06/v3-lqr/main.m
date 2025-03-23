% main.m
clear; clc; close all;

%% System parameters
m    = 50;        % kg
I    = 2.5;       % kg*m^2
g    = 1.62;      % m/s^2
l1   = 0.5;       % m
T_max= 150;       % N

%% Simulation parameters
dt     = 0.05;
T_total= 120;
time   = 0:dt:T_total;

%% Linearized state-space matrices for observer/controller
% State: [x; dx; y; dy; theta; dtheta]
A = [0, 1, 0, 0, 0, 0;
     0, 0, 0, 0, g, 0;
     0, 0, 0, 1, 0, 0;
     0, 0, 0, 0, 0, 0;
     0, 0, 0, 0, 0, 1;
     0, 0, 0, 0, 0, 0];
B = [0, 0;
     0, g;
     0, 0;
     -1/m, 0;
     0, 0;
     0, m*g*l1/I];
% Modified C: Now we measure x, y and theta.
C = [1, 0, 0, 0, 0, 0;
     0, 0, 1, 0, 0, 0;
     0, 0, 0, 0, 1, 0];

%% Noise covariances for LQE (observer design)
Q     = diag([0.1, 0.1, 0.1, 0.1, 0.01, 0.01]); % Process noise covariance
R_cov = diag([0.5, 0.5, 0.05]);                 % Measurement noise covariance
G     = eye(6);  % Process noise input matrix

% Compute observer gain L using LQE (Kalman filter gain for continuous-time systems)
[L_obs, P, E] = lqe(A, G, C, Q, R_cov);

%% LQR Controller Design for Tracking
% Augmented cost matrices for the 6-state system
Q_lqr = diag([10, 0.1, 1, 0.1, 10, 0.1]);
R_lqr = diag([0.1, 0.1]);
% Compute LQR gain K (designed on the linearized model)
[K_lqr, S, E_lqr] = lqr(A, B, Q_lqr, R_lqr);

% Equilibrium input (hover): T_eq = m*g, phi_eq = 0.
u_eq = [m*g; 0];

%% Lander drawing parameters (for both true and estimated)
lander_dims.lander_width   = 0.8;  % meters
lander_dims.lander_height  = 1.2;  % meters
lander_dims.nozzle_length  = 1.0;  % meters

%% Initial conditions
% True state: [x; dx; y; dy; theta; dtheta]
x_true = zeros(6,1);
x_true(3) = 12;   % Initial altitude = 10 m
x_true(4) = -2;

% Observer initial estimate
x_hat = zeros(6,1);
x_hat(3) = 12;
x_hat(4) = -2;

%% Setup figure, sliders, and plot handles
f = figure('Name', 'Lander Simulation with Observer & LQR Controller',...
    'NumberTitle','off','Position',[100 100 1100 600]);

% Create reference sliders
uicontrol('Style','text','Position',[20 550 140 20],'String','Reference x (m)');
r_x_slider = uicontrol('Style','slider','Min',-5,'Max',5,'Value',0,...
    'Position',[20 530 200 20]);
uicontrol('Style','text','Position',[20 500 140 20],'String','Reference Altitude r_y (m)');
r_y_slider = uicontrol('Style','slider','Min',0,'Max',12,'Value',10,...
    'Position',[20 480 200 20]);

% (Optional: You may remove the manual thrust/nozzle sliders in LQR mode)
% For demonstration, we hide them:
% uicontrol('Style','text','Position',[20 450 140 20],'String','Thrust T (N)');
% thrust_slider = uicontrol(...);  % Not used in LQR control
% uicontrol('Style','text','Position',[20 420 140 20],'String','Nozzle Angle \phi (rad)');
% phi_slider = uicontrol(...);      % Not used in LQR control

% Axes for the 2D plot
ax = axes('Position',[0.35 0.1 0.6 0.8]);
hold on; grid on;
xlabel('X Position (m)');
ylabel('Altitude (m)');
xlim([-5 5]); ylim([0 12]);
title('Lander: True state (black) vs Estimated state (blue) vs Reference (red star)');

% Create plot handles for the true and estimated landers and their nozzles,
% and a marker for the reference.
h_true        = plot(0,0, 'ks-', 'LineWidth',2, 'MarkerSize',8, 'MarkerFaceColor','k');
h_nozzle_true = plot(0,0, 'r-', 'LineWidth',2);
h_est         = plot(0,0, 'bo--', 'LineWidth',2, 'MarkerSize',8);
h_nozzle_est  = plot(0,0, 'c--', 'LineWidth',2);
h_ref         = plot(0,0, 'r*', 'MarkerSize',12, 'LineWidth',2);

%% Simulation loop
k = 1;
while ishandle(f) && k <= length(time)
    % Get reference inputs from sliders
    r_x = get(r_x_slider, 'Value');
    r_y = get(r_y_slider, 'Value');
    % Form the reference state (desired [x; dx; y; dy; theta; dtheta])
    x_ref = [r_x; 0; r_y; 0; 0; 0];
    
    % Compute LQR control input based on the observer estimate:
    % u = u_eq - K_lqr*(x_hat - x_ref)
    u = lqrController(x_hat, x_ref, u_eq, K_lqr);
    
    % Update true state using the nonlinear dynamics (Euler integration)
    x_true = updateTrueState(x_true, u, dt, m, I, g, l1);
    
    % Simulate measurement (add slight Gaussian noise)
    noise_level = [0.05; 0.05; 0.01];  % std dev for x, y, theta
    y_meas = C * x_true + noise_level .* randn(3, 1);
    
    % Update observer using the linearized model
    x_hat = updateObserver(x_hat, u, y_meas, dt, A, B, C, L_obs);
    
    % Compute the lander shapes for both the true and estimated states.
    % (The nozzle drawing uses the commanded nozzle deflection, here it is zero at equilibrium.)
    nozzle_command = u(2);  % In our LQR design, we expect phi to be driven toward zero.
    [lander_x_true, lander_y_true, nozzle_x_true, nozzle_y_true] = computeLanderShape(x_true, lander_dims, nozzle_command);
    [lander_x_est,  lander_y_est,  nozzle_x_est,  nozzle_y_est]  = computeLanderShape(x_hat,  lander_dims, nozzle_command);
    
    % Update plot handles for the true state
    set(h_true, 'XData', lander_x_true, 'YData', lander_y_true);
    set(h_nozzle_true, 'XData', nozzle_x_true, 'YData', nozzle_y_true);
    
    % Update plot handles for the estimated state
    set(h_est, 'XData', lander_x_est, 'YData', lander_y_est);
    set(h_nozzle_est, 'XData', nozzle_x_est, 'YData', nozzle_y_est);
    
    % Update reference marker
    set(h_ref, 'XData', r_x, 'YData', r_y);
    
    % Update title with current altitude and orientation for true vs estimated and reference
    title(ax, sprintf('Altitude: True = %.2f m, Est = %.2f m, Ref = %.2f m | Theta: True = %.2f rad, Est = %.2f rad',...
        x_true(3), x_hat(3), r_y, x_true(5), x_hat(5)));
    
    drawnow;
    pause(dt);
    k = k + 1;
end
