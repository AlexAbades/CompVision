function [h]= Gaussian(sigma,l,x)
%GAUSSIAN: Create a gaussian Kernel given the variance. It can be a 1D 
%          Gaussian or a 2D Gaussian.
% sigma-> Variance of the Gaussian.
% l -> Specify 1 or 2. For 1D or 2D Kernel.

if nargin < 3
    x=[-1 0 1];
end
y = x';

if l==1 %1D
    h=(1/sqrt(2*pi*sigma)).*exp(-x.^2/2*sigma);
else %2D
    h=(1/(2*sigma*pi))*exp(-(x.^2+y.^2)/2*sigma);
end

end



%{
    x=[-1 0 1];
    y=[-1 0 1]';
    if l==1 %1D
        h=(1/sqrt(2*pi*sigma)).*exp(-x.^2/2*sigma);
    else %2D
        h=(1/(2*sigma*pi))*exp(-(x.^2+y.^2)/2*sigma);
    end
    
end

%}
