function [x, y] = initializeSnake(I)

% Show figure
figure
imshow(I)

% Get initial points
num_points = 0;
axis on
hold on;
x_points = 0;
y_points = 0;

%break loop on getting enter key
while true
    [x_c,y_c, button] = ginput(1);
    if isempty(x_c) || button(1) == 10; break; end
    num_points = num_points+1;
    x_points(num_points) = x_c;
    y_points(num_points) = y_c;
    plot(x_points(num_points),y_points(num_points),'x')
    drawnow
end
x_points(num_points+1) = x_points(1);
y_points(num_points+1) = y_points(1);
t = linspace(0,1,num_points+1);

% Interpolate
m = 400;
tt = linspace(0,1,m);
x = spline(t,x_points,tt);
y = spline(t,y_points,tt);

% Clamp points to be inside of image
[imh, imw] = size(I);
[x, y] = clampPoints(x,y,imh, imw,m);

% plot initial snake
plot(x,y,'r','LineWidth',1.5);
axis equal
hold off
axis off
end