%% CHECKING GRAPHS: 
sig = 4.5;
s = 3*sig;

x = [-20];
for i=1:0.2:20
    x(end + 1) =  (-20+i);
end
for i=1:0.2:20
    x(end+1) = i;
end

g = Gaussian(sig, 1, x);

figure()
plot(x,g)

% h = (1/sig*sqrt(2*pi)).*exp(-x.^2/2*sigma.^2);
% NOT EQUAL AS IN THE BOOK 


%% IMAGE CONVOLUTION: Gaussian
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


%% Image Convolution: Gaussian Derivative 
% Check the comutative property: First make the derivative of the image
% times the kernel
% Variance: 1

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

%%  Large Gaussian derivative vs cumulative of small Gaussian

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

% Load the txt of coordinates.
X= load('curves/dino_noisy.txt');
D = load('curves/dino.txt');

% Check the image
figure
plot(D(:,1), D(:,2), 'green')
hold on 
plot(X(:,1), X(:,2), 'red')
legend('Dino', 'Dino Noisy')
title('Dinos')
hold off
axis equal

% Extract the dimensions of our matrix given the coordinates.
[m,~] = size(X);

% Create the L matrix with diafgonal [1 2 1] and corners upper right and
% lower left equal 1.

% Identity matrix.
I=eye(m);
% Matrix (n,n) with diagonal with value -2.
L=I*-2;
% Matrix with uper diagonal with values 1.
M = diag(ones(1,m-1),1);
% Matrix with lower diagonal with values equal 1.
M1= diag(ones(1,m-1),-1);
% Sum all matrices to obtain L matrix.
L=L+M+M1;
% Replace the lower left and upper right corners with 1.
L(m,1)=1;L(1,m)=1;

% Assign a value for lambda:
lambda=0.5;

% Calculate the new curve smooth with the average value of the neighbours.
X_new=(I+lambda*L)*X;

% Plot the result against the original
figure()
plot(D(:,1), D(:,2), 'green')
hold on 
plot(X_new(:,1), X_new(:,2), 'blue')
hold on
plot(X(:,1), X(:,2), 'red')
hold on 
legend('Dino', 'Dino 1st curve smooth', 'Noisy Dino')
axis equal
title('Dinos smoothness')
hold off 

% Check different lambdas
lambdas = 0.1:0.1:1;

for i=1:length(lambdas)
    X_temp = (I+lambdas(i)*L)*X;
    
    str = sprintf('lambda value: %u', lambdas(i));
    plot(X_temp(:,1), X_temp(:,2), 'DisplayName', str)
    hold on 
    

end
hold off 
legend show
axis equal
title('Checking different values of lambda')



% Iterate with small values of lambda:
lambda = 0.1;
k = 100;
X_temp = X;
figure()

for i=1:k
    X_temp = (I+lambda*L)*X_temp;
    if ~mod(i,10)
        str = sprintf('Iteration number: %u', i);
        plot(X_temp(:,1), X_temp(:,2), 'DisplayName', str)
        hold on 
    end

end
hold off 
legend show
title('Iterating 100 times')


%% Implicit smothing (avoiding iteration) 
lambda = 5;
X_imp = (I-lambda*L)\X; % \ operator same as the inv()

% Plot the results 
figure
plot(D(:,1), D(:,2), 'green')
hold on 
plot(X_imp(:,1), X_imp(:,2), 'red')
legend('Dino', 'Dino Noisy')
title('Dinos')
hold off


% We can see that for large values of lambda the result is the same as if
% we were iterating many times we also loose some information due to the
% shrinkage of the curve.


%% Elasticity and rigidity:

% Elasticity cte
alpha = 1;
% Rigidity cte 
beta = 1;

% Create the matrix A 
A = L;

% Create the B matrix 
B = L + eye(m)*-4;

% Calculate the new curve smooth
X_er = (I-alpha*A - beta*B)\X;

figure()
plot(X_er(:,1), X_er(:,2))
hold on 


%% Function implementet with the posibility of changeing A and B matrices 


Y = smooothing(X, 2, 1);
P = smooothing(X, 2, 1, [0 1 -2], [0 1 -6]);
figure()
plot(X_er(:,1), X_er(:,2))
hold on 
plot(Y(:,1), Y(:,2))
plot(P(:,1), P(:,2), 'm') % It seems to be an error at the code, cause the 
% plot shows different results, but only at the end of the line. 

%% Quiz 1: Exercise 1

% Read and show Image 
I= mat2gray(imread('noisy_number.png'));
figure()
imshow(I)

% Create a 2D Gaussian Kernel

% Create 2 vectors x and y
sigma = 10;
x = [-1 0 1]; % Does the extended version follow the binomial? 

n1 = ceil(sqrt(sigma));
n2 = ceil(sigma/n1);

figure()
for i=1:sigma
    k = Gaussian(i,2,x);
    n = imfilter(I,k);
    subplot(n1,n2,i)
    imagesc(n)
    colormap('gray')
end

%% Quiz 1: Exercise 2

F1= imread('fuel_cells/fuel_cell_1.tif');

L=length_SB(F1);

%% %% Quiz 1: Exercise 3

% Load the txt of coordinates.
X= load('curves/dino_noisy.txt');
D = load('curves/dino.txt');

% Extract the dimensions of our matrix given the coordinates.
[m,~] = size(X);

% Create the L matrix with diafgonal [1 2 1] and corners upper right and
% lower left equal 1.

I=eye(m);
L=I*-2;
M = diag(ones(1,m-1),1);
M1= diag(ones(1,m-1),-1);
L=L+M+M1;
L(m,1)=1;L(1,m)=1;

% Assign a value for lambda:
lambda=0.25;

% Calculate the new curve smooth with the average value of the neighbours.
X_new=(I+lambda*L)*X; % Coordinates 

curveLength = sum(vecnorm(diff(X_new),2,2));


