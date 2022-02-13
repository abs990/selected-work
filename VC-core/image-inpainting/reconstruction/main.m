clc;
clear;
close all;

imgin = im2double(imread('./large.jpg'));

[imh, imw, nb] = size(imgin);
assert(nb==1);
% the image is grayscale

V = zeros(imw, imh);
V(1:imh*imw) = 1:imh*imw;
% V(y,x) = (y-1)*imw + x
% use V(y,x) to represent the variable index of pixel (x,y)
% Always keep in mind that in matlab indexing starts with 1, not 0

%TODO: initialize counter, A (sparse matrix) and b.
%-----
%compute number of equations needed
additional_constraints = 0;
num_equations = imh * imw + additional_constraints;
%compute array length to store index of non-zero entries for sparse matrix
% Non-zero coefficient count -  
% 5 for non-edge, 3 for edge and non-corner, 1 for corner
num_nonzero_indexes = 5*(imh-2)*(imw-2)+3*(2*imh-4)+3*(2*imw-4)+4;
%initiliase i,j,k,b,counter
i = zeros(1,num_nonzero_indexes);
j = zeros(1,num_nonzero_indexes);
k = zeros(1,num_nonzero_indexes);
b = zeros(num_equations,1);
counter=1;
%for non-edge pixels
for c = 2:(imh-1)
    for d = 2:(imw-1)
        %fill coefficient entries needed for equation for pixel (x,y)
        i(1,counter:counter+4) = V(d,c);
        j(1,counter:counter+4) = [V(d,c) V(d,c+1) V(d,c-1) V(d+1,c) V(d-1,c)];
        k(1,counter:counter+4) = [4 -1 -1 -1 -1];
        counter = counter+5;
        %compute entry for b
        sg_up = imgin(c,d) - imgin(c,d-1);
        sg_down = imgin(c,d) - imgin(c,d+1);
        sg_left = imgin(c,d) - imgin(c-1,d);
        sg_right = imgin(c,d) - imgin(c+1,d);
        b(V(d,c)) = sg_up + sg_down + sg_left + sg_right;
    end    
end
%for pixels on left and right edge (non-corner)
for c = 2:(imh-1)
    %coefficient entries for left edge
    i(1,counter:counter+2) = V(1,c);
    j(1,counter:counter+2) = [V(1,c) V(1,c+1) V(1,c-1)];
    k(1,counter:counter+2) = [2 -1 -1];
    counter = counter + 3;
    %coefficient entries for right edge
    i(1,counter:counter+2) = V(imw,c);
    j(1,counter:counter+2) = [V(imw,c) V(imw,c+1) V(imw,c-1)];
    k(1,counter:counter+2) = [2 -1 -1];
    counter = counter + 3;
    %entries in b for left egde
    sg_up = imgin(c,1) - imgin(c-1,1);
    sg_down = imgin(c,1) - imgin(c+1,1);
    b(V(1,c)) = sg_up + sg_down;
    %entries in b for right edge
    sg_up = imgin(c,imw) - imgin(c-1,imw);
    sg_down = imgin(c,imw) - imgin(c+1,imw);
    b(V(imw,c)) = sg_up + sg_down;
end
%for pixels on top or bottom edge (non-corner)
for c = 2:(imw-1)
    %coefficient entries for top edge
    i(1,counter:counter+2) = V(c,1);
    j(1,counter:counter+2) = [V(c,1) V(c+1,1) V(c-1,1)];
    k(1,counter:counter+2) = [2 -1 -1];
    counter = counter + 3;
    %coefficient entries for bottom edge
    i(1,counter:counter+2) = V(c,imh);
    j(1,counter:counter+2) = [V(c,imh) V(c+1,imh) V(c-1,imh)];
    k(1,counter:counter+2) = [2 -1 -1];
    counter = counter + 3;
    %entries in b for top edge
    sg_left = imgin(1,c) - imgin(1,c-1);
    sg_right = imgin(1,c) - imgin(1,c+1);
    b(V(c,1)) = sg_left + sg_right;
    %entries in b for bottom edge
    sg_left = imgin(imh,c) - imgin(imh,c-1);
    sg_right = imgin(imh,c) - imgin(imh,c+1);
    b(V(c,imh)) = sg_left + sg_right;
end    
%-----
%TODO: add extra constraints
%-----
i(counter:counter+3) = [V(1,1) V(imw,1) V(1,imh) V(imw,imh)];
j(counter:counter+3) = [V(1,1) V(imw,1) V(1,imh) V(imw,imh)];
k(counter:counter+3) = 1;
counter = counter + 3;

%adjust constraints as required
b(V(1,1)) = imgin(1,1);
b(V(imw,1)) = imgin(1,imw);
b(V(1,imh)) = imgin(imh,1);
b(V(imw,imh)) = imgin(imh,imw);
%-----
%TODO: fill the elements in A for each pixel in the image
%-----
A = sparse(i,j,k);
%size(A)
%size(b)
%-----

%TODO: solve the equation
%use "lscov" or "\", please google the matlab documents
solution = lscov(A,b);
error = sum(abs(A*solution-b));
disp(error)
imgout = (reshape(solution,[imw,imh]))';

imwrite(imgout,'output.png');
figure(), hold off, imshow(imgout);
