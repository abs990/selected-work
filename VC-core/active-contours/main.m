clear;
clc;

% Parameters (play around with different images and different parameters)
%for all except star
N = 1000;alpha = 1.4;beta = 0.2;gamma = 0.5;
kappa = 0.15;Wline = 0.5;Wedge = 1.0;Wterm = 0.5;sigma = 0.5;
%for star
%N = 1000;alpha = 0.7;beta = 0.1;gamma = 0.8;
%kappa = 0.15;Wline = -0.5;Wedge = 1.0;Wterm = 0.5;sigma = 0.8;

% Load image
I = imread('images/circle.jpg');
%I = imread('images/square.jpg');
%I = imread('images/shape.png');
%I = imread('images/star.png');
%I = imread('images/vase.tif');
%I = imread('images/dental.png');
%I = imread('images/brain.png');

if (ndims(I) == 3)
    I = rgb2gray(I);
end

% Initialize the snake
[x, y] = initializeSnake(I);
pause(2);

% Calculate external energy
I_smooth = im2double(imgaussfilt(I, sigma));
Eext = getExternalEnergy(I_smooth,Wline,Wedge,Wterm);

% Calculate matrix A^-1 for the iteration
%Ainv = getInternalEnergyMatrix(size(x,2), alpha, beta, gamma);
Ainv = getInternalEnergyMatrixBonus(size(x,2), alpha, beta, gamma);

% Iterate and update positions
numDisplays = N/100;
displaySteps = floor(N/numDisplays);
pause_display_time = 0.0001;
for i=1:N
    % Iterate
    [x,y] = iterate(Ainv, x, y, Eext, gamma, kappa);

    % Plot intermediate result
    imshow(I);
    axis on
    hold on;
    plot([x x(1)], [y y(1)], 'r', 'LineWidth', 2.5);
        
    % Display step
    if(mod(i,displaySteps)==0)
        %plot([x x(1)], [y y(1)], 'r', 'LineWidth', 2.5);
        fprintf('%d/%d iterations\n',i,N);
        pause(2);
    end
    
    pause(pause_display_time)
end
 
if(displaySteps ~= N)
    fprintf('%d/%d iterations\n',N,N);
end