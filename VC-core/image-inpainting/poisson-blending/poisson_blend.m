function imgout = poisson_blend(im_s, mask_s, im_t)
% -----Input
% im_s     source image (object)
% mask_s   mask for source image (1 meaning inside the selected region)
% im_t     target image (background)
% -----Output
% imgout   the blended image

[imh, imw, nb] = size(im_s);

%TODO: consider different channel numbers

%TODO: initialize counter, A (sparse matrix) and b.
%Note: A don't have to be k¡Ák,
%      you can add useless variables for convenience,
%      e.g., a total of imh*imw variables
%-----
V = zeros(imw, imh);
num_pixels = imh*imw;
V(1:num_pixels) = 1:num_pixels; % V(y,x) = (y-1)*imw + x. Use V(y,x) to represent the variable index of pixel (x,y)

%max number of non-zero entries possible in the coefficient matrix.
num_nonzero_indexes = 5*(imh-2)*(imw-2)+3*(2*imh-4)+3*(2*imw-4)+4;

%i,j,v required for constructing A. One list for each channel.
%based on space used, these matrices will be shortened later
i = zeros(nb,num_nonzero_indexes); 
j = zeros(nb,num_nonzero_indexes);
v = zeros(nb,num_nonzero_indexes);
b = zeros(num_pixels,nb);

for ch = 1:nb
    counter = 1;
    for c = 1:imh
        for d = 1:imw
            if(mask_s(c,d) == 1)
            % compute coefficients
                % coefficient for pixel(c,d)
                i(ch,counter) = V(d,c);
                j(ch,counter) = V(d,c);
                v(ch,counter) = 4;
                counter = counter + 1;
                % coefficient for pixel (c,d-1) if needed
                if(d~=1 && mask_s(c,d-1) == 1)
                    i(ch,counter) = V(d,c);
                    j(ch,counter) = V(d-1,c);
                    v(ch,counter) = -1;
                    counter = counter + 1;
                end    
                % coefficient for pixel (c,d+1) if needed
                if(d~=imw && mask_s(c,d+1) == 1)
                    i(ch,counter) = V(d,c);
                    j(ch,counter) = V(d+1,c);
                    v(ch,counter) = -1;
                    counter = counter + 1;
                end 
                % coefficient for pixel (c-1,d) if needed
                if(c~=1 && mask_s(c-1,d) == 1)
                    i(ch,counter) = V(d,c);
                    j(ch,counter) = V(d,c-1);
                    v(ch,counter) = -1;
                    counter = counter + 1;
                end 
                % coefficient for pixel (c+1,d) if needed
                 if(c~=imh && mask_s(c+1,d) == 1)
                    i(ch,counter) = V(d,c);
                    j(ch,counter) = V(d,c+1);
                    v(ch,counter) = -1;
                    counter = counter + 1;
                 end
            %compute corresponding value for b
                 s_value = 4 * im_s(c,d,ch);                
                 if(d~=1)
                     s_value = s_value - im_s(c,d-1,ch);
                 end
                 if(d~=imw)
                     s_value = s_value - im_s(c,d+1,ch);
                 end
                 if(c~=1)
                     s_value = s_value - im_s(c-1,d,ch);
                 end
                 if(c~=imh)
                     s_value = s_value - im_s(c+1,d,ch);
                 end
                 
                 %update neighbours based on presence within mask
                 if(d~=1 && mask_s(c,d-1) == 0) 
                     s_value = s_value + im_t(c,d-1,ch); 
                 end
                 if(d~=imw && mask_s(c,d+1) == 0) 
                     s_value = s_value + im_t(c,d+1,ch); 
                 end
                 if(c~=1 && mask_s(c-1,d) == 0) 
                     s_value = s_value + im_t(c-1,d,ch); 
                 end
                 if(c~=imh && mask_s(c+1,d) == 0) 
                     s_value = s_value + im_t(c+1,d,ch); 
                 end
                 b(V(d,c),ch) = s_value; %put final value in b
            else
                % for pixels outside of the mask
                i(ch,counter) = V(d,c);
                j(ch,counter) = V(d,c);
                v(ch,counter) = 1;
                counter = counter + 1;
                b(V(d,c),ch) = im_t(c,d,ch);
            end    
        end    
    end
end

%slice out just the populated non-zero entries in i,j,v
num_nonzero_indexes = counter-1;
i = i(:,1:num_nonzero_indexes);
j = j(:,1:num_nonzero_indexes);
v = v(:,1:num_nonzero_indexes);
%-----

%TODO: fill the elements in A and b, for each pixel in the image
%solve equation for each channel
%-----
solution = zeros(num_pixels,nb);
for channel = 1:nb
    sparseA = sparse(i(channel,:),j(channel,:),v(channel,:));
    solution(:,channel) = lscov(sparseA,b(:,channel));
    error = sum(abs(sparseA*solution(:,channel)-b(:,channel)));
    disp(error);
end    
%-----

%TODO: copy those variable pixels to the appropriate positions
%      in the output image to obtain the blended image
%-----
imgout = zeros(imh,imw,nb);
for channel = 1:nb
    imgout(:,:,channel) = (reshape(solution(:,channel),[imw,imh]))';
end   
%-----