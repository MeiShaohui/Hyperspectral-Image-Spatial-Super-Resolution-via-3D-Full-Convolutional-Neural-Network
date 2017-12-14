import os
import h5py



import network3d

import argparse
import os
#os.environ['CUDA_VISIBLE_DEVICES']='0'


def train():

    model = network3d.srcnn()
    model.summary()


    h5f = h5py.File(args.input_data, 'r')
    X = h5f['data']
    y = h5f['label'].value[:,:,:,4:-4,:]

    n_epoch = args.n_epoch

    if not os.path.exists(args.save):
        os.mkdir(args.save)

    for epoch in range(0, n_epoch,10):
        model.fit(X, y, batch_size=16, nb_epoch=10, shuffle='batch')
        if args.save:
            print("Saving model ", epoch + 10)
            model.save(os.path.join(args.save, 'model_%d.h5' %(epoch+10)))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-S', '--save',
                        default='./save',
                        dest='save',
                        type=str,
                        nargs=1,
                        help="Path to save the checkpoints to")
    parser.add_argument('-D', '--data',
                        default='data_process/pa_train_3d_all_data.h5',
                        dest='input_data',
                        type=str,
                        nargs=1,
                        help="Training data directory")
    parser.add_argument('-E', '--epoch',
                        default=200,
                        dest='n_epoch',
                        type=int,
                        nargs=1,
                        help="Training epochs must be a multiple of 5")
    args = parser.parse_args()
    print(args)
    train()
