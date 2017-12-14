clear;close all;
%% settings
% folder = 'Train';
savepath = 'pavia_train.h5';
size_input = 33;
size_label = 21;
scale = 2;

stride = 5;
snr=60;

%% initialization
dataa = zeros(size_input, size_input, 110, 1,1);
label = zeros(size_label, size_label, 110, 1,1);
padding = abs(size_input - size_label)/2;
count = 0;

%% generate data


load('pa_mirror9.mat')
image_old = data(:,:,:);
image_or=image_old;
%  [o,p,q] = size(image_old);
%   for b = 1:q
%       image_old(:,:,b) = image_old(:,:,b).*(1+randn(o,p)/(snr)); % Add Gaussian noise
%   end



%% 
tic

    
    image = image_old(:,:,:);
    image_true=image_or(:,:,:);
    %image = rgb2ycbcr(image);
    image = im2double(image(:, :, :));
    
    im_label = modcrop(image, scale);
    image_true=modcrop(image_true, scale);
    
    [hei,wid,l] = size(im_label);
    im_input = gaussian_down_sample(im_label,scale);
%   im_input = imresize(imresize(im_label,1/scale,'bicubic'),[hei,wid],'bicubic');
    im_input = imresize(im_input,[hei,wid],'bicubic');

   % disp(i)
    for x = 1 : stride : hei-size_input+1
        disp(x )
        for y = 1 :stride : wid-size_input+1
            
            subim_input = im_input(x : x+size_input-1, y : y+size_input-1,:);
            subim_label = image_true(x+padding : x+padding+size_label-1, y+padding : y+padding+size_label-1,:);

            count=count+1;
            dataa(:, :, :,1, count) = subim_input;
            label(:, :, :, 1,count) = subim_label;
        end
    end


order = randperm(count);
dataa = dataa(:, :, :, order);
label = label(:, :, :, order); 

%% writing to HDF5
chunksz = 128;
created_flag = false;
totalct = 0;

for batchno = 1:floor(count/chunksz)
    last_read=(batchno-1)*chunksz;
    batchdata = dataa(:,:,:,last_read+1:last_read+chunksz); 
    batchlabs = label(:,:,:,last_read+1:last_read+chunksz);
    disp([batchno, floor(count/chunksz)])
    startloc = struct('dat',[1,1,1,totalct+1], 'lab', [1,1,1,totalct+1]);
    curr_dat_sz = store2hdf5(savepath, batchdata, batchlabs, ~created_flag, startloc, chunksz); 
    created_flag = true;
    totalct = curr_dat_sz(end);
end
toc
h5disp(savepath);
