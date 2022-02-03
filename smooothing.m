function [X_final] = smooothing(X,alpha, beta, a, b)
%SMOOTHING implicit smoothing with the extended kernel.
%   X is the coordinate matrix, matrix that stores the coordinates of the
%   line (x,y)
%   alpha: Elasticity variable
%   beta: Rigidity
%   A, B: list of the elments we want into the matrix, it must be from left to
%   right as in the matrix it's from bottom to the top. It will make it
%   symetric

% Get size of X
[m,~] = size(X);
% Identity matrix.
I=eye(m);

if nargin <= 3
    % Matrix A
    
    % Matrix (n,n) with diagonal with value -2.
    L=I*-2;
    % Matrix with uper diagonal with values 1.
    M = diag(ones(1,m-1),1);
    % Matrix with lower diagonal with values equal 1.
    M1= diag(ones(1,m-1),-1);
    % Sum all matrices to obtain L matrix.
    A=L+M+M1;
    % Replace the lower left and upper right corners with 1.
    A(m,1)=1;
    A(1,m)=1;
    
    % Matrix B by default: Same procediment as A.
    L=I*-6;
    M = diag(ones(1,m-1),1);
    M1= diag(ones(1,m-1),-1);
    B=L+M+M1;
    B(m,1)=1;
    B(1,m)=1;


elseif nargin == 4 
    % Create a square matrix of dimensions m, and value 0
    A = zeros(m,m);
    % Foor loop over list A to specify the bottom diagonals of the matrix
    % and sum over all the matrices created to obtain the combination.
    for i=length(a):-1:1
        cont = length(a)-i;
        A = A + diag(ones(m-cont,1),-cont)*a(i);
    end
    
    % Make the matrix symetric
    A = triu(A.',1) + tril(A);
    

    % Matrix B by default
    I=eye(m);
    L=I*-6;
    M = diag(ones(1,m-1),1);
    M1= diag(ones(1,m-1),-1);
    B=L+M+M1;
    B(m,1)=1;
    B(1,m)=1;

elseif nargin == 5

    % Matrix A
    A = zeros(m,m);
    for i=length(a):-1:1
        cont = length(a)-i;
        A = A + diag(ones(m-cont,1),-cont)*a(i);
    end
    A = triu(A.',1) + tril(A);

    % Matrix B
    B = zeros(m,m);
    for i=length(b):-1:1
        cont = length(b)-i;
        B = B + diag(ones(m-cont,1),-cont)*b(i);
    end
    B = triu(B.',1) + tril(B);

end

X_final = (I-alpha*A - beta*B)\X;







end
