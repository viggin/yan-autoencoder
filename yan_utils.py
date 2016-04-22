"""
Yet ANother autoencoder toolbox.

Utility functions.

Copyright 2016 YAN Ke, Tsinghua Univ. http://yanke23.com , xjed09@gmail.com

"""

import numpy as np
import scipy.optimize as opt
import theano
import theano.tensor as tensor
import time


def tea(size=(3, 3)):
    """Return a simple 2D or 3D array.

    :param size: the size of the array to generate.

    """
    n_dims = len(size)
    assert 2 <= n_dims <= 3, "only supports 2D or 3D array"

    len1 = np.prod(size[0:2])
    a = np.reshape(np.arange(len1)+1, size[0:2])
    if n_dims == 2:
        return a
    elif n_dims == 3:
        adder = np.reshape(np.arange(size[2])*10, (-1,1,1))
        return a + adder


def gauss_fun_tensor(x):
    """Gaussian function for tensor x."""

    return tensor.exp(-x ** 2)


def act_fun_from_name(act_fun_name):
    """Return the tensor activation function according to act_fun_name"""

    act_funs = {
        'lin':  None,
        'linear':  None,
        'sigm': tensor.nnet.sigmoid,
        'sigmoid': tensor.nnet.sigmoid,
        'tanh': tensor.tanh,
        'relu': tensor.nnet.relu,
        'softmax':  tensor.nnet.softmax_op,
        'sm':   tensor.nnet.softmax_op,
        'gauss':    gauss_fun_tensor
    }
    return act_funs[act_fun_name.lower()]


def loss_fun_from_name(loss_fun_name):
    """Return the tensor loss function according to loss_fun_name"""

    loss_funs = {
        'mse': lambda x, y: tensor.mean((x-y)**2, axis=1),  # mean squared error
        'mae': lambda x, y: tensor.mean(abs(x-y), axis=1),  # mean absolute error

        # log loss: for binary features in reconstruction by autoencoders?
        # y should be in [0,1], or nan will occur. Not for softmax!
        'log': lambda x, y: -tensor.mean(x*tensor.log(y) +
                                         (1-x)*tensor.log(1-y), axis=1),

        # cross-entropy loss, for comparing softmax predictions and true targets
        'ce': lambda x, y: -tensor.mean(x * tensor.log(y), axis=1),
    }
    return loss_funs[loss_fun_name.lower()]


def opt_name_from_abbrv(opt_name_abbrv):
    """Return the full name of the optimization algorithm according to the abbreviation"""

    opt_names = {
        'mb': 'minibatch',
        'minib': 'minibatch',
        'minibatch': 'minibatch',
        'cg': 'cg',
        'lb': 'l-bfgs-b',
        'l-bfgs-b': 'l-bfgs-b',
        'b': 'bfgs',
        'bfgs': 'bfgs',
    }
    return opt_names[opt_name_abbrv.lower()]


def make_one_hot_target(label_vector, max_label=None):
    """Use the one-hot (dummy variable) scheme to encode label_vector

    :param label_vector: start from 1
    :param max_label: maximum number of classes
    :return labelVector.shape[0]-by-maxLabel array

    """
    if max_label is None:
        max_label = label_vector.max()
    tar = np.zeros((label_vector.shape[0], max_label), dtype=theano.config.floatX)
    tar[range(label_vector.shape[0]), label_vector - 1] = 1
    return tar


def save2txt(num_list, filename):
    with open(filename, 'r+') as f:
        n_lines = int(f.readline())
        n_lines += 1
        f.seek(0)
        f.write(('%d\n' % n_lines))

    with open(filename, 'a') as f:
        f.write(('\n\t%d\t' % n_lines))
        f.write(time.strftime('(%H:%M:%S %y-%m-%d %a)')+'\t\t\n')
        f.writelines('%.4f\t' % num for num in num_list)
        # f.write('\n')


def pdist(samples_x, samples_y=None, dist_func=None):
    """Return the pairwise distance matrix, each row of the inputs is a sample"""

    if samples_y is None:
        samples_y = samples_x
    if dist_func is None:
        dist_func = lambda x, y: ((x - y) ** 2).sum()

    dist_mat = np.zeros((samples_x.shape[0], samples_y.shape[0]))
    for i in range(samples_x.shape[0]):
        for j in range(samples_y.shape[0]):
            dist_mat[i, j] = dist_func(samples_x[i, :], samples_y[j, :])

    return dist_mat

if __name__ == '__main__':
    print tea((3, 3, 3))
