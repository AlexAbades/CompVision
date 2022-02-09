function [d_h]= Derivative_Gaussian(sigma)
    x=[-1 0 1];
    d_h=(-x/(sigma^3*sqrt(2*pi))).*exp(-x.^2/2*(sigma^2));
end

