clear;close all;
size_input = 1096;
size_label = 715;
scale = 2;
scale2 = 6;
stride = 14;
snr=30;
l=1096;
w=714;
d=110;

%% initialization
 dataa = zeros(l, w, d);
 label = zeros(l, w, d);
% padding = abs(size_input - size_label)/2;
% count = 0;


 load('pa_mirror9.mat');

  
 image_ori = data(:,:,:);
 image_old = data(:,:,:);
%  
%       [o,p,q] = size(image_old);
%         for b = 1:q
%              image_old(:,:,b) = image_old(:,:,b).*(1+randn(o,p)/(snr)); % Add Gaussian noise relative to radiance
%          end
% % % % %  
% 


    
    image = image_old(:,:,:);
    %image = im2double(image(:, :, :));   
    im_label = modcrop(image, scale);   
    [hei,wid,l] = size(im_label);
    im_input = gaussian_down_sample(im_label,scale);

    im_input = imresize(im_input,[hei,wid],'bicubic');
    
    %im_input = imresize(im_input,[hei,wid],'bilinear');
    %im_input = imresize(im_input,[hei,wid],'nearest');
    
    image_o = image_ori(:,:,:);
    im_true  = modcrop(image_o, scale);
  
     dataa(:, :, :) = im_input;
     label(:, :, :) = im_true;
   
  
 save 'pavia_train' dataa label
 
