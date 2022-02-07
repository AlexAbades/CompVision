function [L,m]= length_SB(Image)
% LENGTH_SB: Creates a mask given an image. It plots the original and the
% mask. In addition, computes the sum of all the boundaries. Returns the
% length and the mask.

% Create a mask of the Image. 
m=boundarymask(Image);

% Creates a figure and displays the images. 
figure(1)
subplot(1,2,1)
imshow(Image)
title('Original Image')
subplot(1,2,2)
imshow(m)
title('Boundary Image')

% Computes the length of the the Boundaries of the mask
L=sum(sum(m));

end

    
    

