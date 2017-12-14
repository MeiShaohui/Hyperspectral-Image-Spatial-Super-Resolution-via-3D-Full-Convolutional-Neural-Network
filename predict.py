from os import listdir
from os.path import isfile, join
import argparse
import h5py
import math

import numpy as np

from sklearn.metrics import  mean_squared_error

import network3d

import matplotlib.pyplot as plt

import time
import scipy.io as sio
from keras import backend as K

input_size = 33
label_size = 21
pad = (33 - 21) // 2


class SRCNN(object):
    def __init__(self, weight):
        self.model = network3d.srcnn((None, None,None,1))
        self.model.summary()

        f = h5py.File(weight, mode='r')
        self.model.load_weights_from_hdf5_group(f['model_weights'])

    def predict(self, data, **kwargs):
        use_3d_input = kwargs.pop('use_3d_input', True)
        if use_3d_input :

            im_out = [self.model.predict(data)]

        else:
            im_out = [self.model.predict(data)]
            if data.ndim != 2:
                raise ValueError('the dimension of data must be 2 !!')
            im_out = self.model.predict(data[None, :, :, None])
        return np.asarray(im_out)

def show_picture(data):
    plt.imshow(data,plt.cm.gray)
    plt.show()

def test_for_all_bands(input,label):

    input_new=np.zeros([1,input.shape[0],input.shape[1],input.shape[2],1])
    label_new=np.zeros([1,label.shape[0],label.shape[1],label.shape[2],1])

    for i in range(input.shape[2] ):
        input_new[0, :, :, i,0] = input[:, :, i]
        label_new[0, :, :, i,0] = label[:, :, i]


    return input_new, label_new[:,:,:,4:-4,:]

def predict():

    srcnn = SRCNN(option.model)
    f = sio.loadmat('data_process/data/pa_test.mat')
    input=f['dataa'].astype(np.float32)
    label=f['label'].astype(np.float32)
    print  input.shape
    print  label.shape


    psnr(label[ 6:-6, 6:-6, 4:-4], input[6:-6, 6:-6, 4:-4])
    ssim(label[ 6:-6, 6:-6, 4:-4], input[6:-6, 6:-6, 4:-4])
    sam(label[ 6:-6, 6:-6, 4:-4], input[6:-6, 6:-6, 4:-4])

    input,label=test_for_all_bands(input,label)

    print  input.shape
    start = time.time()
    output = srcnn.predict(input[:,:,:,:,:])
    end = time.time()
    print "the time is : ", end-start
    print label.shape
    print output.shape


    show_picture(output[0,0,:, :, 25,0 ])
    show_picture(input[0,  :, :, 25, 0])
    show_picture(label[0,  :, :, 25, 0])
    print '123'


    psnr(label[0,6:-6,6:-6,:,0],output[0,0,:,:,:,0])
    ssim(label[0,6:-6,6:-6,:,0],output[0,0,:,:,:,0])
    sam(label[0,6:-6,6:-6,:,0],output[0,0,:,:,:,0])
    #f = h5py.File('save_pavia_result', 'w')
    #f.create_dataset(name='input', data=input)
    #f.create_dataset(name='output', data=output)
    #f.create_dataset(name='label', data=label)
    #f.close()


def psnr(x_true, x_pred):

    print x_true.shape
    print x_pred.shape
    n_bands = x_true.shape[2]
    PSNR = np.zeros(n_bands)
    MSE = np.zeros(n_bands)
    mask = np.ones(n_bands)
    x_true=x_true[:,:,:]
    for k in range(n_bands):
        x_true_k = x_true[  :, :,k].reshape([-1])
        x_pred_k = x_pred[  :, :,k,].reshape([-1])

        MSE[k] = mean_squared_error(x_true_k, x_pred_k, )


        MAX_k = np.max(x_true_k)
        if MAX_k != 0 :
            PSNR[k] = 10 * math.log10(math.pow(MAX_k, 2) / MSE[k])
            #print ('P', PSNR[k])
        else:
            mask[k] = 0

    psnr = PSNR.sum() / mask.sum()
    mse = MSE.mean()
    print('psnr', psnr)
    print('mse', mse)
    return psnr, mse

def ssim(x_true,x_pre):

    num=x_true.shape[2]
    ssimm=np.zeros(num)
    c1=0.0001
    c2=0.0009
    n=0
    for x in range(x_true.shape[2]):


            z = np.reshape(x_pre[:, :,x], [-1])
            sa=np.reshape(x_true[:,:,x],[-1])
            y=[z,sa]
            cov=np.cov(y)
            oz=cov[0,0]
            osa=cov[1,1]
            ozsa=cov[0,1]
            ez=np.mean(z)
            esa=np.mean(sa)
            ssimm[n]=((2*ez*esa+c1)*(2*ozsa+c2))/((ez*ez+esa*esa+c1)*(oz+osa+c2))
            n=n+1
    SSIM=np.mean(ssimm)
    print ('SSIM',SSIM)

def sam(x_true,x_pre):
    print x_pre.shape
    print x_true.shape
    num = (x_true.shape[0]) * (x_true.shape[1])
    samm = np.zeros(num)
    n = 0
    for x in range(x_true.shape[0]):
        for y in range(x_true.shape[1]):
            z = np.reshape(x_pre[ x, y,:], [-1])
            sa = np.reshape(x_true[x, y,:], [-1])
            tem1=np.dot(z,sa)
            tem2=(np.linalg.norm(z))*(np.linalg.norm(sa))
            samm[n]=np.arccos(tem1/tem2)
            n=n+1
    SAM=(np.mean(samm))*180/np.pi
    print('SAM',SAM)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-M', '--model',

                        default='model/model_pa.h5',
                        dest='model',
                        type=str,
                        nargs=1,
                        help="The model to be used for prediction")

    option = parser.parse_args()


    predict()
