
%% IMAGE CONVOLUTION
I= mat2gray(imread('fibres_xcth.png'));

%Filters
x=[-1 0 1];
y=[-1 0 1]';

h1=Gaussian(1,1);
h2=Gaussian(1,2);


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

sigma=1;
d_h1=Derivative_Gaussian(sigma);


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

h5 = Gaussian(sigma1,2);
g5=imfilter(I,h5);

sigma2=2;
h6= Gaussian(sigma2,2);
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

%%  Large Gaussian derivative vs

%Large derivative Gaussian
sigma=20;
h8=Derivative_Gaussian(sigma);
g8=imfilter(I,h8);


%Gaussian with t = 10 and a Gaussian derivative with t = 10
sigma=10;
h9=Gaussian(sigma,2);
g9=imfilter(I,h8);

h10=Derivative_Gaussian(sigma);
g10=imfilter(g9,h10);


figure(3)
subplot(1,2,1)
imagesc(g8)
colormap(gray)
colorbar
title('Large derivative Gaussian,t=20')

subplot(1,2,2)
imagesc(g10)
colormap(gray)
colorbar
title('Gaussian+Gaussian derivative,t = 10')


%% SEGMENTATION BOUNDARY

F1= imread('fuel_cells/fuel_cell_1.tif');

L=length_SB(F1);




%% CURVE SMOOTHING
X= load('curves/dino_noisy.txt');
%L matriz
[m,~] = size(X);

L=zeros(m,m);
u=ones(1,m);
L=diag(u)*-2;
M = diag(ones(1,m-1),1);
M1= diag(ones(1,m-1),-1);
L=L+M+M1;
L(m,1)=1;L(1,m)=1;


I=eye(m);
lambda=0.5;

X_new=(I-lambda.*L).*X;






