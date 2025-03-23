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
T_total= 20;
time   = 0:dt:T_total;

%% Linearized state-space matrices for observer
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

%% Noise covariances for LQE
Q     = diag([0.1, 0.1, 0.1, 0.1, 0.01, 0.01]); % Process noise covariance
R_cov = diag([0.5, 0.5, 0.05]);                 % Measurement noise covariance
G     = eye(6);  % Assume identity process noise input

% Compute observer gain L using LQE (Kalman filter gain for continuous-time systems)
[L, P, E] = lqe(A, G, C, Q, R_cov);

%% Lander drawing parameters (for both true and estimated)
lander_dims.lander_width   = 0.8;  % meters
lander_dims.lander_height  = 1.2;  % meters
lander_dims.nozzle_length  = 1.0;  % meters

%% Initial conditions
% True state: [x; dx; y; dy; theta; dtheta]
x_true = zeros(6,1);
x_true(3) = 10;   % Altitude = 10 m

% Observer initial estimate
x_hat = zeros(6,1);
x_hat(3) = 10;

%% Setup figure, sliders, and plot handles
f = figure('Name', 'Lander Simulation with Observer','NumberTitle','off','Position',[100 100 1000 600]);

% Thrust slider
uicontrol('Style','text','Position',[20 550 120 20],'String','Thrust T (N)');
thrust_slider = uicontrol('Style','slider','Min',0,'Max',T_max,'Value',m*g,...
    'Position',[20 530 200 20]);

% Nozzle angle slider
uicontrol('Style','text','Position',[20 500 120 20],'String','Nozzle Angle \phi (rad)');
phi_slider = uicontrol('Style','slider','Min',-0.2,'Max',0.2,'Value',0,...
    'Position',[20 480 200 20]);

% Axes for the 2D plot
ax = axes('Position',[0.35 0.1 0.6 0.8]);
hold on; grid on;
xlabel('X Position (m)');
ylabel('Altitude (m)');
xlim([-5 5]); ylim([0 12]);
title('Lander: True state (black) vs Estimated state (blue)');

% Create plot handles for true and estimated landers and their nozzles
h_true        = plot(0,0, 'ks-', 'LineWidth',2, 'MarkerSize',8, 'MarkerFaceColor','k');
h_nozzle_true = plot(0,0, 'r-', 'LineWidth',2);
h_est         = plot(0,0, 'bo--', 'LineWidth',2, 'MarkerSize',8);
h_nozzle_est  = plot(0,0, 'c--', 'LineWidth',2);

%% Simulation loop
k = 1;
while ishandle(f) && k <= length(time)
    % Get control inputs from sliders
    T_slider = get(thrust_slider, 'Value');
    phi      = get(phi_slider, 'Value');  % Nozzle deflection (rad)
    
    % User slider = "Throttle" adjustment (intuitive: up means rise)
    T_hover = m * g;
    delta_T = T_slider - T_hover;  % Positive delta_T = "more throttle"
    
    % Map it properly for the nonlinear model
    T = T_hover - delta_T;   % More slider => less T => accelerates upward
    u = [T; phi];
    
    % Update true state using the nonlinear dynamics (Euler integration)
    x_true = updateTrueState(x_true, u, dt, m, I, g, l1);
    
    % Simulate measurement (here we assume no noise, but noise can be added)
    noise_level = [0.05; 0.05; 0.01];  % Noise standard deviation for x, y, theta
    y_meas = C * x_true + noise_level .* randn(3, 1);
    
    % Update observer (Luenberger observer) using the linearized model
    x_hat = updateObserver(x_hat, u, y_meas, dt, A, B, C, L);
    
    % Compute the lander shapes for both the true and estimated states.
    [lander_x_true, lander_y_true, nozzle_x_true, nozzle_y_true] = computeLanderShape(x_true, lander_dims, phi);
    [lander_x_est,  lander_y_est,  nozzle_x_est,  nozzle_y_est]  = computeLanderShape(x_hat,  lander_dims, phi);
    
    % Update plot handles for the true state
    set(h_true, 'XData', lander_x_true, 'YData', lander_y_true);
    set(h_nozzle_true, 'XData', nozzle_x_true, 'YData', nozzle_y_true);
    
    % Update plot handles for the estimated state
    set(h_est, 'XData', lander_x_est, 'YData', lander_y_est);
    set(h_nozzle_est, 'XData', nozzle_x_est, 'YData', nozzle_y_est);
    
    % Update title with current altitude and orientation for true vs estimated
    title(ax, sprintf('Altitude: True = %.2f m, Est = %.2f m | Theta: True = %.2f rad, Est = %.2f rad',...
        x_true(3), x_hat(3), x_true(5), x_hat(5)));
    
    drawnow;
    pause(dt);
    k = k+1;
end
