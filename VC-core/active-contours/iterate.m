function [newX, newY] = iterate(Ainv, x, y, Eext, gamma, kappa)

% Get fx and fy
[Eext_x, Eext_y] = imgradientxy(Eext);  

% Iterate
[imh, imw] = size(Eext);
Eext_snake_x = interp2(Eext_x,x,y);
Eext_snake_y = interp2(Eext_y,x,y);
newX = Ainv * (gamma * x' + kappa * Eext_snake_x');
newY = Ainv * (gamma * y' + kappa * Eext_snake_y');

% Clamp to image size
[newX, newY] = clampPoints(newX', newY', imh, imw, size(x,2));
end
