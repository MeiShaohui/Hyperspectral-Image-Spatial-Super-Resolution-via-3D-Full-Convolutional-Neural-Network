import scipy.io as sio
import h5py
import numpy as np

def turn_to_train(filename):
    
    file = h5py.File(filename)
    data = file['data']
    label = file['label']

    print(data.shape)
    print(label.shape)
    data = np.swapaxes(data, 1, 2)
    data = np.swapaxes(data, 2, 3)
    data = np.reshape(data, [data.shape[0], data.shape[1], data.shape[2], data.shape[3], 1])
    print(data.shape)

    label = np.swapaxes(label, 1, 2)
    label = np.swapaxes(label, 2, 3)
    label = np.reshape(label, [label.shape[0], label.shape[1], label.shape[2], label.shape[3], 1])
    print(label.shape)

    f = h5py.File('pa_train_3d_all_data.h5', 'w')
    f.create_dataset('data', data=data)
    f.create_dataset('label', data=label)

if __name__ == '__main__':
     turn_to_train('pavia_train.h5')
