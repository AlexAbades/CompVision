
function [L]= length_SB(Image)

m=boundarymask(Image);

figure(1)
subplot(1,2,1)
imshow(Image)
title('Original Image')
subplot(1,2,2)
imshow(m)
title('Boundary Image')


L=sum(sum(m));

end

    
    

