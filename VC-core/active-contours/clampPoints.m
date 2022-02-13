function[newX, newY] = clampPoints(x,y,imh, imw,numPoints)

newX = zeros(1,numPoints);
newY = zeros(1,numPoints);

for i = 1:numPoints
    % clamp x co-ordinates
    if x(i) <= 0
        newX(i) = 1;
    elseif x(i) >= imw
        newX(i) = imw-1;
    else
        newX(i) = x(i);
    end
    % clamp y co-ordinates
    if y(i) <= 0
        newY(i) = 1;
    elseif y(i) >= imh
        newY(i) = imh-1;
    else
        newY(i) = y(i);
    end    
end

end