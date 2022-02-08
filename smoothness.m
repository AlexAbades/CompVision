function [d] = smoothness(I)
%SMOOTHNESS: Calculates the total variation of an image.

d = sum(sum(abs(diff(I,1,2)))) + sum(sum(abs(diff(I,2,1))));

end