function [lander_x, lander_y, nozzle_x, nozzle_y] = computeLanderShape(x_state, dims, phi)
% computeLanderShape: Compute polygon coordinates for the lander body and the nozzle.
%   x_state: state vector [x; dx; y; dy; theta; dtheta]
%   dims: structure with fields lander_width, lander_height, nozzle_length
%   phi: nozzle deflection angle (rad)
%
% Returns:
%   lander_x, lander_y: coordinates for the lander body polygon
%   nozzle_x, nozzle_y: coordinates for the nozzle line (2 points)

% Unpack state
x_pos = x_state(1);
y_pos = x_state(3);
theta = x_state(5);

% Define lander rectangle in body frame (centered at origin)
w = dims.lander_width;
h = dims.lander_height;
lander_shape = [ -w/2,  w/2,  w/2, -w/2, -w/2;
                 -h/2, -h/2,  h/2,  h/2, -h/2];

% Rotation matrix to convert from body to world coordinates.
% Here we use R = [cos(theta) sin(theta); -sin(theta) cos(theta)]
% so that a positive theta produces the expected visual tilt.
R = [cos(theta), sin(theta); -sin(theta), cos(theta)];

% Rotate the lander shape and translate it
lander_rot = R * lander_shape;
lander_x = lander_rot(1,:) + x_pos;
lander_y = lander_rot(2,:) + y_pos;

% Compute the nozzle (attached at the bottom center of the lander in the body frame)
nozzle_base_body = [0; -h/2];
% In the body frame, define the nozzle direction using phi (for phi = 0 the nozzle points straight down)
nozzle_dir_body = [sin(phi); -cos(phi)];
nozzle_tip_body = nozzle_base_body + dims.nozzle_length * nozzle_dir_body;

% Rotate and translate the nozzle points
nozzle_base_rot = R * nozzle_base_body;
nozzle_tip_rot  = R * nozzle_tip_body;
nozzle_x = [nozzle_base_rot(1) + x_pos, nozzle_tip_rot(1) + x_pos];
nozzle_y = [nozzle_base_rot(2) + y_pos, nozzle_tip_rot(2) + y_pos];
end
