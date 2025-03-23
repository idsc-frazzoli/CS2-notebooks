% System characteristics
m = 50;        % kg
I = 2.5;       % kg*m^2
g = 1.62;      % m/s^2
l_1 = 0.5;     % m
T_max = 150;   % N

% State-Space matrices (linearized about hover)
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
     0, m*g*l_1/I];
C = [1, 0, 0, 0, 0, 0;
     0, 1, 0, 0, 0, 0];

% Simulation setup
dt = 0.05;        % Time step (s)
T_total = 20;     % Total simulation time (s)
time = 0:dt:T_total;

% Initial state [x, dx, y, dy, theta, dtheta]
X = zeros(6,1);
X(3) = 10;  % Initial altitude = 10 m

% Create figure and sliders
f = figure('Name','Lander Manual Control','NumberTitle','off','Position',[200 200 800 600]);

% Thrust slider
uicontrol('Style','text','Position',[50 550 120 20],'String','Thrust T (N)');
thrust_slider = uicontrol('Style','slider',...
    'Min',0,'Max',T_max,'Value',m*g,...  % Hover thrust = m*g
    'Position',[50 530 200 20]);

% Nozzle angle slider
uicontrol('Style','text','Position',[50 500 120 20],'String','Nozzle Angle \phi (rad)');
phi_slider = uicontrol('Style','slider','Min',-0.5,'Max',0.5,'Value',0,...
    'Position',[50 480 200 20]);

% Plot setup
ax = axes('Position',[0.4,0.2,0.55,0.7]);
hold on; grid on;
xlabel('X Position (m)');
ylabel('Altitude (m)');
xlim([-5 5]); ylim([0 12]);
title('Lander Trajectory');

% Plot handles for the lander body and nozzle
lander_body = plot(0, 0, 'ks-', 'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', 'k');
nozzle_plot = plot(0, 0, 'r-', 'LineWidth', 2);

% Define lander and nozzle dimensions (for drawing)
lander_width = 0.8;   % meters
lander_height = 1.2;  % meters
nozzle_length = 1.0;  % meters

% Simulation loop with real-time interaction
k = 1;
while ishandle(f) && k <= length(time)
    % Get manual control inputs from sliders
    T_slider = get(thrust_slider, 'Value');
    phi = get(phi_slider, 'Value');
    
    % Compute delta_T (relative to hover); mapping flipped so that
    % increasing the slider (more thrust) makes the lander rise.
    delta_T = m*g - T_slider;
    u = [delta_T; phi];
    
    % Update state using Euler integration
    X = X + dt*(A*X + B*u);
    
    % Extract current state values
    x_pos = X(1);
    altitude = X(3);  % Altitude (y positive upward)
    theta = X(5);     % Lander orientation (radians)
    phi_val = phi;    % Nozzle deflection (radians)
    
    %% Update Lander Body Drawing
    % Define the lander rectangle in the body frame.
    % In the drawing, the body frame has x right and y up.
    % With theta = 0, the lander is upright with its "nozzle" at the bottom.
    lander_shape = [ -lander_width/2,  lander_width/2,  lander_width/2, -lander_width/2, -lander_width/2;
                     -lander_height/2, -lander_height/2, lander_height/2,  lander_height/2, -lander_height/2 ];
    
    % Use a rotation matrix with -theta so that the drawn orientation matches the dynamics.
    % (R = [cos(-theta) -sin(-theta); sin(-theta) cos(-theta)] is equivalent to 
    %  [cos(theta) sin(theta); -sin(theta) cos(theta)] ).
    R = [cos(-theta), -sin(-theta); sin(-theta), cos(-theta)];
    
    % Rotate the lander shape and translate to the current position.
    lander_rotated = R * lander_shape;
    lander_x = lander_rotated(1,:) + x_pos;
    lander_y = lander_rotated(2,:) + altitude;
    set(lander_body, 'XData', lander_x, 'YData', lander_y);
    
    %% Update Nozzle Drawing
    % Attach the nozzle at the bottom of the lander (in body frame the bottom is at y = -lander_height/2).
    nozzle_base_body = [0; -lander_height/2];
    % Define the nozzle direction in the body frame.
    % For phi = 0 the nozzle points straight down (i.e. in the negative y direction).
    nozzle_dir_body = [sin(phi_val); -cos(phi_val)];  
    nozzle_tip_body = nozzle_base_body + nozzle_length * nozzle_dir_body;
    
    % Rotate nozzle base and tip to world coordinates and translate.
    nozzle_base_world = R * nozzle_base_body + [x_pos; altitude];
    nozzle_tip_world = R * nozzle_tip_body + [x_pos; altitude];
    set(nozzle_plot, 'XData', [nozzle_base_world(1), nozzle_tip_world(1)], ...
                     'YData', [nozzle_base_world(2), nozzle_tip_world(2)]);
    
    % Update title with current altitude, orientation, and nozzle deflection.
    title(ax, sprintf('Altitude = %.2f m | \\theta = %.2f rad | \\phi = %.2f rad', altitude, theta, phi_val));
    
    drawnow;
    pause(dt);
    k = k + 1;
end
