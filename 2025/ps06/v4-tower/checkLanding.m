function [landed, success, vel] = checkLanding(x, dims, safe_zone, hazard_zone)
% checkLanding: Determines if the lander has landed and whether it is a safe landing.
%
% Landing is triggered if:
%  - Any vertex of the lander touches the ground (y <= 0), or
%  - Any vertex touches the hazard zone (black tower).
%
% A safe landing is declared if:
%  - The entire lander is within the safe landing zone (green square),
%  - AND the horizontal (dx) and vertical (dy) speeds are below 0.2 m/s.
%
% Inputs:
%   x         - state vector [x; dx; y; dy; theta; dtheta]
%   dims      - lander drawing parameters (used to compute the polygon)
%   safe_zone - structure with fields: x, y, width, height for the safe zone
%   hazard_zone - structure with fields: x, y, width, height for the hazard zone
%
% Outputs:
%   landed  - Boolean, true if a landing event is detected.
%   success - Boolean, true if the landing is safe.
%   vel     - [dx; dy] velocities at landing.
%
% This function uses the lander body polygon (computed with nozzle deflection = 0).

vel_threshold = 0.2;  % m/s

% Compute the lander polygon (body) using computeLanderShape (ignore nozzle deflection)
[px, py, ~, ~] = computeLanderShape(x, dims, 0);

% Get horizontal and vertical velocities
vel = [x(2); x(4)];

landed = false;
success = false;

% Condition 1: Any vertex touches the ground (y <= 0)
if any(py <= 0)
    landed = true;
end

% Condition 2: Any vertex touches the hazard zone (black tower)
for i = 1:length(px)
    if inZone(px(i), py(i), hazard_zone)
        landed = true;
        success = false;
        return;  % Immediate failure if hazard is touched
    end
end

% Condition 3: Check if the entire lander is within the safe zone (green square)
all_in_safe = all(arrayfun(@(i) inZone(px(i), py(i), safe_zone), 1:length(px)));

if all_in_safe
    landed = true;
    % If fully inside safe zone, check velocities for a soft landing
    if (abs(vel(1)) < vel_threshold) && (abs(vel(2)) < vel_threshold)
        success = true;
    else
        success = false;
    end
end

% If landed but not entirely in safe zone, then it is a crash (failure)
if landed && ~all_in_safe
    success = false;
end

end

function flag = inZone(xp, yp, zone)
% inZone: Returns true if the point (xp, yp) lies within the defined zone.
flag = (xp >= zone.x) && (xp <= zone.x + zone.width) && ...
       (yp >= zone.y) && (yp <= zone.y + zone.height);
end
