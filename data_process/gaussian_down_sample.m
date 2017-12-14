function HSI = gaussian_down_sample(data,w)
%------------------------------------------------------------------
% This function downsample HS image with Gaussian point spread function
%
% HSI = gaussian_down_sample(data,w)
%
% INPUT
%       data            : input HS image (xdata,ydata,band)
%       w               : multiple difference of ground sampling distance (scalar)
%
% OUTPUT
%       HSI             : downsampled HS image (band, xdata/gsd, ydata/gsd)
%
%------------------------------------------------------------------

[xdata, ydata, band] = size(data);
hx = floor(xdata/w); hy = floor(ydata/w);
HSI = zeros(hx, hy, band);
k = 2.7725887/w^2;
if mod(w,2)==0
    [X, Y] = meshgrid(-(w-1)/2:1:(w-1)/2, -(w-1)/2:1:(w-1)/2);
    gauss1 = exp(-(X.^2+Y.^2)*k);
    gauss1 = gauss1/sum(gauss1(:)); % Normalized Gaussian distribution within wxw size window
    [X, Y] = meshgrid(-(2*w-1)/2:1:(2*w-1)/2, -(2*w-1)/2:1:(2*w-1)/2);
    gauss2 = exp(-(X.^2+Y.^2)*k);
    gauss2 = gauss2/sum(gauss2(:)); % Normalized Gaussian distribution within 2wx2w size window
    for x = 1:hx
        for y = 1:hy
            if x==1 || x==hx || y==1 || y==hy
                HSI(x,y,:) = sum(sum(double(data(1+(x-1)*w:w+(x-1)*w,1+(y-1)*w:w+(y-1)*w,:)).*repmat(gauss1,[1 1 band]),1),2);
            else
                HSI(x,y,:) = sum(sum(double(data(1+(x-1)*w-w/2:w+(x-1)*w+w/2,1+(y-1)*w-w/2:w+(y-1)*w+w/2,:)).*repmat(gauss2,[1 1 band]),1),2);
            end
        end
    end
else
    [X, Y] = meshgrid(-(w-1)/2:1:(w-1)/2, -(w-1)/2:1:(w-1)/2);
    gauss1 = exp(-(X.^2+Y.^2)*k);
    gauss1 = gauss1/sum(gauss1(:)); % Normalized Gaussian distribution within wxw size window
    [X, Y] = meshgrid(-(w-1):1:w-1, -(w-1):1:w-1);
    gauss2 = exp(-(X.^2+Y.^2)*k);
    gauss2 = gauss2/sum(gauss2(:)); % Normalized Gaussian distribution within (2w-1)x(2w-1) size window
    for x = 1:hx
        for y = 1:hy
            if x==1 || x==hx || y==1 || y==hy
                HSI(x,y,:) = sum(sum(double(data(1+(x-1)*w:w+(x-1)*w,1+(y-1)*w:w+(y-1)*w,:)).*repmat(gauss1,[1 1 band]),1),2);
            else
                HSI(x,y,:) = sum(sum(double(data(1+(x-1)*w-(w-1)/2:w+(x-1)*w+(w-1)/2,1+(y-1)*w-(w-1)/2:w+(y-1)*w+(w-1)/2,:)).*repmat(gauss2,[1 1 band]),1),2);
            end
        end
    end
end