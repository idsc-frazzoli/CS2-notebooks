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
% Modified C: measure x, y, and theta.
C = [1, 0, 0, 0, 0, 0;
     0, 0, 1, 0, 0, 0;
     0, 0, 0, 0, 1, 0];

%% Noise covariances for LQE (observer design)
Q     = diag([0.1, 0.1, 0.1, 0.1, 0.01, 0.01]);
R_cov = diag([0.5, 0.5, 0.05]);
G     = eye(6);

% Compute observer gain L using LQE
[L_obs, P, E] = lqe(A, G, C, Q, R_cov);

%% LQR Controller Design for Tracking
% (Tuning cost matrices for this simulation)
Q_lqr = diag([1, 0.1, 10, 0.1, 10, 0.1]);
R_lqr = diag([0.05, 0.1]);
[K_lqr, S, E_lqr] = lqr(A, B, Q_lqr, R_lqr);

% Equilibrium (hover) input: T_eq = m*g, phi_eq = 0.
u_eq = [m*g; 0];

%% Lander drawing parameters
lander_dims.lander_width   = 0.8;  % meters
lander_dims.lander_height  = 1.2;  % meters
lander_dims.nozzle_length  = 1.0;  % meters

%% Initial conditions
% True state: [x; dx; y; dy; theta; dtheta]
x_true = zeros(6,1);
x_true(3) = 12;   % Initial altitude = 12 m
x_true(4) = -2 + rand(1, 1);   % Initial vertical velocity

% Observer initial estimate
x_hat = zeros(6,1);
x_hat(3) = 12;
x_hat(4) = -2 + rand(1,1);

%% Define Landing Zones
% Safe landing zone (green square): from x=3 to 4, y=0 to 1.
safe_zone.x = 3.5;
safe_zone.y = 1;
safe_zone.width = 1;
safe_zone.height = 1.5;

% Hazard zone (black tower): to the right of safe zone, e.g., x=4 to 4.5, y=0 to 3.
hazard_zone.x = safe_zone.x + safe_zone.width; % starting at x = 4
hazard_zone.y = 0;
hazard_zone.width = 0.5;
hazard_zone.height = 3;

%% Setup figure, sliders, and plot handles
f = figure('Name', 'Lander Simulation with Observer, LQR & Gamification',...
    'NumberTitle','off','Position',[100 100 1100 600]);

% Create reference sliders for r_x and r_y:
uicontrol('Style','text','Position',[20 570 140 20],'String','Reference x (m)');
r_x_slider = uicontrol('Style','slider','Min',-5,'Max',5,'Value',0,...
    'Position',[20 550 200 20]);
uicontrol('Style','text','Position',[20 520 140 20],'String','Reference Altitude r_y (m)');
r_y_slider = uicontrol('Style','slider','Min',0,'Max',12,'Value',10,...
    'Position',[20 500 200 20]);

% Axes for the simulation plot
ax = axes('Position',[0.35 0.1 0.6 0.8]);
hold on; grid on;
xlabel('X Position (m)');
ylabel('Altitude (m)');
xlim([-5 5]); ylim([0 12]);
title('Lander: True (black) vs Estimated (blue) vs Reference (red star)');

% Draw the safe landing zone (green square)
safe_x = [safe_zone.x, safe_zone.x+safe_zone.width, safe_zone.x+safe_zone.width, safe_zone.x];
safe_y = [safe_zone.y, safe_zone.y, safe_zone.y+safe_zone.height, safe_zone.y+safe_zone.height];
h_safe = patch(safe_x, safe_y, 'g', 'FaceAlpha', 0.5, 'EdgeColor', 'k');

% Draw the hazard zone (black tower)
hazard_x = [hazard_zone.x, hazard_zone.x+hazard_zone.width, hazard_zone.x+hazard_zone.width, hazard_zone.x];
hazard_y = [hazard_zone.y, hazard_zone.y, hazard_zone.y+hazard_zone.height, hazard_zone.y+hazard_zone.height];
h_hazard = patch(hazard_x, hazard_y, 'k', 'FaceAlpha', 0.5, 'EdgeColor', 'none');

% Create plot handles for lander bodies, nozzles, and reference marker
h_true        = plot(0,0, 'ks-', 'LineWidth',2, 'MarkerSize',8, 'MarkerFaceColor','k');
h_nozzle_true = plot(0,0, 'r-', 'LineWidth',2);
h_est         = plot(0,0, 'bo--', 'LineWidth',2, 'MarkerSize',8);
h_nozzle_est  = plot(0,0, 'c--', 'LineWidth',2);
h_ref         = plot(0,0, 'r*', 'MarkerSize',12, 'LineWidth',2);

%% Simulation loop
k = 1;
landed = false;
while ishandle(f) && k <= length(time)
    % Get reference inputs from sliders
    r_x = get(r_x_slider, 'Value');
    r_y = get(r_y_slider, 'Value');
    % Form reference state: desired [x; dx; y; dy; theta; dtheta]
    x_ref = [r_x; 0; r_y; 0; 0; 0];
    
    % Compute LQR control based on the observer estimate:
    % u = u_eq - K_lqr*(x_hat - x_ref)
    u = lqrController(x_hat, x_ref, u_eq, K_lqr);
    
    % Update true state using the nonlinear dynamics
    x_true = updateTrueState(x_true, u, dt, m, I, g, l1);
    
    % Simulate measurement (with slight Gaussian noise)
    noise_level = [0.05; 0.05; 0.01];
    y_meas = C * x_true + noise_level .* randn(3,1);
    
    % Update observer using the linearized model
    x_hat = updateObserver(x_hat, u, y_meas, dt, A, B, C, L_obs);
    
    % Compute lander shapes for drawing (nozzle command is u(2); expected near zero)
    nozzle_command = u(2);
    [lander_x_true, lander_y_true, nozzle_x_true, nozzle_y_true] = computeLanderShape(x_true, lander_dims, nozzle_command);
    [lander_x_est,  lander_y_est,  nozzle_x_est,  nozzle_y_est]  = computeLanderShape(x_hat, lander_dims, nozzle_command);
    
    % Update plot handles for the true state
    set(h_true, 'XData', lander_x_true, 'YData', lander_y_true);
    set(h_nozzle_true, 'XData', nozzle_x_true, 'YData', nozzle_y_true);
    
    % Update plot handles for the estimated state
    set(h_est, 'XData', lander_x_est, 'YData', lander_y_est);
    set(h_nozzle_est, 'XData', nozzle_x_est, 'YData', nozzle_y_est);
    
    % Update reference marker
    set(h_ref, 'XData', r_x, 'YData', r_y);
    
    % Check landing conditions using the new function.
    [landed, success, vel] = checkLanding(x_true, lander_dims, safe_zone, hazard_zone);
    if landed
        if success
            result_str = sprintf('SUCCESS! Landed on safe zone. dx = %.2f m/s, dy = %.2f m/s', vel(1), vel(2));
        else
            result_str = sprintf('FAILURE! Crash landing. dx = %.2f m/s, dy = %.2f m/s', vel(1), vel(2));
        end
        title(ax, result_str);
        break;  % End simulation loop upon landing
    else
        % Update title with ongoing state info
        title(ax, sprintf('Altitude: True = %.2f m, Est = %.2f m, Ref = %.2f m | Theta: True = %.2f rad, Est = %.2f rad',...
            x_true(3), x_hat(3), r_y, x_true(5), x_hat(5)));
    end
    
    drawnow;
    pause(dt);
    k = k + 1;
end

if landed
    if success
        disp('Landing SUCCESS!');
    else
        disp('Landing FAILURE!');
    end
    fprintf('Final velocities: dx = %.2f m/s, dy = %.2f m/s\n', vel(1), vel(2));
else
    disp('Simulation ended without landing.');
end
