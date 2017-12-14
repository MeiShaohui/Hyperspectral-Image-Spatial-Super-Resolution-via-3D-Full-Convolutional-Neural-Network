import scipy.io as sio
import h5py
import numpy as np



def get_data(filename):
    data = sio.loadmat(filename)
    data = data['pavia']

    return data

def get_mirror(data):
    data_new = np.concatenate((np.reshape(data[:,:,0],[data.shape[0],data.shape[1],1]),data),axis=2)
    data_new = np.concatenate((np.reshape(data[:,:,1],[data.shape[0],data.shape[1],1]),data_new),axis=2)
    data_new = np.concatenate((np.reshape(data[:,:,2],[data.shape[0],data.shape[1],1]),data_new),axis=2)
    data_new = np.concatenate((np.reshape(data[:,:,3],[data.shape[0],data.shape[1],1]),data_new),axis=2)

    data_new = np.concatenate((data_new,np.reshape(data[:,:,-1],[data.shape[0],data.shape[1],1])),axis=2)
    data_new = np.concatenate((data_new,np.reshape(data[:,:,-2],[data.shape[0],data.shape[1],1])),axis=2)
    data_new = np.concatenate((data_new,np.reshape(data[:,:,-3],[data.shape[0],data.shape[1],1])),axis=2)
    data_new = np.concatenate((data_new,np.reshape(data[:,:,-4],[data.shape[0],data.shape[1],1])),axis=2)

    print  data_new.shape
    return data_new

def to_1(data):
    max, min = data.max(), data.min()
    data_new = (data - min) / (max - min)
    return data_new


if __name__ == '__main__':
    filename='data/Pavia.mat'
    data=get_data(filename)
    data=get_mirror(data)
    data=to_1(data)
    sio.savemat('pa_mirror9.mat', {'data': data}) 
  


