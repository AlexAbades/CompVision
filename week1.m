
%IMAGE CONVOLUTION
I= mat2gray(imread('fibres_xcth.png'));

%Filters
x=[-1 0 1];
y=[-1 0 1]';
h1=(1/sqrt(2*pi))*exp(-x.^2/2); 

h2=(1/(2*pi))*exp(-(x.^2+y.^2)/2);


g2=imfilter(I,h2); %2D

g1=imfilter(I,h1);
g11=imfilter(g1,h1');

figure(1)
subplot(2,2,1)
imagesc(I)
colormap(gray)
colorbar
title('Original Image')

subplot(2,2,2)
imagesc(g2)
colormap(gray)
colorbar
title('Convolution 2D Kernel')

subplot(2,2,3)
imagesc(g11)
colormap(gray)
colorbar
title('Convolution 2 orthogonal 1D Kernel')

subplot(2,2,4)
diff=g2-g11; 
imagesc(diff)
colormap(gray)
colorbar
title('2D kernel vs. 1D kernel')


%%
%derivative of the image
%We are taking into account a variance of 1

d=[0.5 0 -0.5];
Id=imfilter(I,d); 
 
g3= imfilter(Id,h1);


d_h1=(-x/sqrt(2*pi)).*exp(-x.^2/2);

g4=imfilter(I,d_h1);

figure(2)
subplot(1,3,1)
imagesc(g3)
colormap(gray)
colorbar
title('Derivative of the Image-Gauss')

subplot(1,3,2)
imagesc(g4)
colormap(gray)
colorbar
title('Image - Derivative of the Gauss')

subplot(1,3,3)
diff=g3-g4; 
imagesc(diff)
colormap(gray)
colorbar
title('Difference')

%% Semigroup structures

sigma1=20;

h5 = (1/(2*sigma1*pi))*exp(-(x.^2+y.^2)/2*sigma1);
g5=imfilter(I,h5);

sigma2=2;
h6 = (1/(2*sigma2*pi))*exp(-(x.^2+y.^2)/2*sigma2);
g6=imfilter(h6,h6);

for i=1:8
   g6=conv2(g6,h6);
end

g7=imfilter(I,g6);


figure(3)
subplot(1,2,1)
imagesc(g5)
colormap(gray)
colorbar
title('t=20')

subplot(1,2,2)
imagesc(g7)
colormap(gray)
colorbar
title('t=2 x 10')






