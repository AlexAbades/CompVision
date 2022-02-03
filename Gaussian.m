function [h]= Gaussian(sigma,l)
    x=[-1 0 1];
    y=[-1 0 1]';
    if l==1 %1D
        h=(1/sqrt(2*pi*sigma)).*exp(-x.^2/2*sigma);
    else %2D
        h=(1/(2*sigma*pi))*exp(-(x.^2+y.^2)/2*sigma);
    end
    
end

