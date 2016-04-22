"""
Yet Another Neural network toolbox.

An implementation of shallow autoencoder using the Theano library.

ref: http://deeplearning.net/tutorial/contents.html

Please cite: Ke Yan, and David Zhang, "Correcting Instrumental Variation
and Time-varying Drift: A Transfer Learning Approach with Autoencoders,"
accepted by Instrumentation and Measurement, IEEE Transactions on

Copyright 2016 YAN Ke, Tsinghua Univ. http://yanke23.com , xjed09@gmail.com

"""

# naming conventions:
# lower_with_under: packages, functions, variables
# CapWords: classes, exceptions
# CAPS_WITH_UNDER: Global/Class Constants

import os
import sys
import timeit
import copy

import numpy
import matplotlib.pyplot as plt
import theano
import theano.tensor as tensor  # import the tensor module
from theano.tensor.shared_randomstreams import RandomStreams  # import the RandomStreams class
import scipy.optimize as opt

from yan_utils import act_fun_from_name, loss_fun_from_name, opt_name_from_abbrv


class YanAe(object):
    """Yet ANother shallow autoencoder.

    Improved from http://deeplearning.net/tutorial/dA.html
    The denoising, sparsity, L1/L2 regularization, and weight tying strategies can be set.

    """

    def __init__(self,
                 n_visible=128,
                 n_hidden=6,
                 act_fun_name_vis='sigm',  # see func act_fun_from_name in yan_utils.py
                 act_fun_name_hid='sigm',
                 corruption_level=.3,  # if >0, use the denoising strategy
                 sparsity_reg=.0,  # sparsity regularization parameter. If >0, use the sparsity strategy
                 sparsity_target=.05,  # rho, see the model of sparsity AE.
                                       # Should be normalized if use tanh kernel. Watch the
                                       # log expressions in case of NaNs (kernels with negative outputs)
                 L1_reg=.00,  # weight of regularization term
                 L2_reg=.00,
                 use_tied_weight=True,
                 use_biases=True,  # if false, do not use bias vectors
                 loss_fun_name='mse',  # loss function name, can use 'log' for binary features. see function
                                       # loss_fun_from_name in yan_utils.py
                 rng=None,  # random number generator, used when initializing the weights
                            # Set this var when you want to control the randomness of the network
                 ):

        self.numpy_rng = (numpy.random.RandomState(6799) if rng is None else rng)
        self.theano_rng = RandomStreams(self.numpy_rng.randint(2 ** 30))  # for denoising AE
        self.act_fun_vis = act_fun_from_name(act_fun_name_vis)
        self.act_fun_hid = act_fun_from_name(act_fun_name_hid)
        self.loss_fun = loss_fun_from_name(loss_fun_name)
        self.use_tied_weight = use_tied_weight

        def init_W(obj, size):
            initial_W = numpy.asarray(
                obj.numpy_rng.uniform(
                    low=-4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                    high=4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                    size=size
                ),  # can't be init to zero! or optimization can't go on
                dtype=theano.config.floatX
            )
            if obj.act_fun_hid == tensor.nnet.sigmoid:
                initial_W *= 4
            return initial_W

        initial_W = init_W(self, size=(n_visible, n_hidden))
        self.W = theano.shared(value=initial_W, name='W', borrow=True)
        if not use_tied_weight:
            initial_Wprime = init_W(self, size=(n_hidden, n_visible))
            self.W_prime = theano.shared(value=initial_Wprime, name='W\'', borrow=True)
        else:
            self.W_prime = self.W.T

        self.params = [self.W]
        if not use_tied_weight:
            self.params.append(self.W_prime)

        if use_biases:
            initial_bvis = numpy.zeros(n_visible, dtype=theano.config.floatX)
            initial_bhid = numpy.zeros(n_hidden, dtype=theano.config.floatX)

            self.b = theano.shared(
                    value=initial_bvis,
                    borrow=True, name='bVis',
                )
            self.b_prime = theano.shared(
                    value=initial_bhid,
                    borrow=True, name='bHid'
                )
            self.params.extend([self.b, self.b_prime])
        else:
            self.b = None
            self.b_prime = None

        self.n_hidden = n_hidden
        self.n_visible = n_visible
        self.use_tied_weight = use_tied_weight
        self.use_biases = use_biases
        self.corruption_level = corruption_level
        self.sparsity_reg = sparsity_reg
        self.sparsity_target = sparsity_target
        self.L1_reg = L1_reg
        self.L2_reg = L2_reg

    def _get_corrupted_input_tensor(self, input_tensor, corruption_level):
        return self.theano_rng.binomial(size=input_tensor.shape, n=1,
                                        p=1 - corruption_level,
                                        dtype=theano.config.floatX) * input_tensor

    def _get_hidden_values_tensor(self, input_tensor):
        """Computes the values of the hidden layer"""

        val = tensor.dot(input_tensor, self.W)
        if self.use_biases: val += self.b
        if self.act_fun_hid is not None:
            val = self.act_fun_hid(val)
        return val

    def _get_reconstructed_input_tensor(self, hidden):
        """Computes the reconstructed input given the values of the hidden layer"""

        val = tensor.dot(hidden, self.W_prime)
        if self.use_biases:
            val += self.b_prime
        if self.act_fun_vis is not None:
            val = self.act_fun_vis(val)
        return val

    def _compute_funs_for_pretrain(self, is_minibatch, learn_rate=.2):
        """

        :param is_minibatch: if true, use minibatch gradient descent; else use
            other optimization algorithms in scipy
        :param learn_rate: only useful if use minibatch

        """
        x = tensor.matrix(name='x', dtype=theano.config.floatX)
        tilde_x = self._get_corrupted_input_tensor(x, self.corruption_level)
        y = self._get_hidden_values_tensor(tilde_x)
        z = self._get_reconstructed_input_tensor(y)
        loss = self.loss_fun(x, z)

        L1 = abs(self.W).sum() + abs(self.W_prime).sum()
        L2 = (self.W_prime**2).sum() + (self.W_prime**2).sum()
        cost_tensor = tensor.mean(loss) \
                      + self.L1_reg * L1 \
                      + self.L2_reg * L2

        if self.sparsity_reg > 0:
            rho_hat = y.mean(axis=0)
            rho = self.sparsity_target
            if self.act_fun_hid == act_fun_from_name('tanh'):
                rho_hat = rho_hat/2+.5
            sparsity_term = tensor.sum(rho * tensor.log(rho / rho_hat) + (1 - rho) * tensor.log((1 - rho) / (1 - rho_hat)))
            cost_tensor += sparsity_term * self.sparsity_reg

        cost_derive_tensor = tensor.grad(cost_tensor, self.params)  # derivative

        if is_minibatch:
            indices = tensor.lvector('indices')  # indices of samples to a minibatch
            updates = [
                (param, param - learn_rate * gparam)
                for param, gparam in zip(self.params, cost_derive_tensor)
            ]

            self.pretrainFun = theano.function(
                inputs=[indices],
                outputs=cost_tensor,
                updates=updates,
                givens={
                    x: self.x_train[indices]
                }
            )
        else:  # use scipy optimization
            self.cost_fun = theano.function(
                    inputs=[],
                    outputs=cost_tensor,
                    givens={
                        x:  self.x_train
                    }
                )

            self.cost_derive_fun = theano.function(
                    inputs=[],
                    outputs=cost_derive_tensor,
                    givens={
                        x:  self.x_train
                    }
                )

        data = tensor.matrix()
        self.hidden_output_fun = theano.function(
                inputs=[data],
                outputs=self._get_hidden_values_tensor(data),
            )

    def pretrain(self,
                 x,  # training data, each row is a sample
                 max_iter=50,
                 opt_method='CG',  # see opt_name_from_abbrv in yan_utils.py
                 learn_rate=0.2, batch_size=20,  # only for minibatch method
                 show=[False, False, False]  # [running msg, optimization msg, cost plot]
                 ):

        if show[0]:
            print 'compiling the shallow autoencoder..'
        self.x_train = theano.shared(name='x', value=x, borrow=True)
        opt_method = opt_name_from_abbrv(opt_method)
        self._compute_funs_for_pretrain(is_minibatch=(opt_method == 'minibatch'),
                                        learn_rate=learn_rate)
        if show[0]:
            print(('training the shallow autoencoder with %d samples and '
                   '%d features using %s..') % (x.shape[0], x.shape[1], opt_method))
        start_time = timeit.default_timer()

        if opt_method == 'minibatch':
            costs = self.pretrain_minibatch(max_iter, learn_rate, batch_size, show[1])
        else:
            costs = self.pretrain_scipy(max_iter, opt_method, show[1])

        end_time = timeit.default_timer()

        if show[0]:
            print 'The optimization for file ' + os.path.split(__file__)[1] + \
              ' ran for %.1fs' % (end_time - start_time)
        if show[2]:
            plt.figure('cost per epoch')
            plt.plot(range(len(costs)), costs)
            # plt.show()

    def pretrain_scipy(self, max_iter=50, opt_method='CG', show=False):

        def theta2Wb(theta_value):
            """Transform var in scipy opt (theta) to var in AE (W and b)"""
            W_len = self.n_visible*self.n_hidden
            self.W.set_value(theta_value[:W_len].
                             reshape((self.n_visible, self.n_hidden)), borrow=True)
            pos = W_len
            if not self.use_tied_weight:
                self.W_prime.set_value(theta_value[W_len:W_len*2].
                                       reshape((self.n_hidden, self.n_visible)),
                                       borrow=True)
                pos = W_len*2
            if self.use_biases:
                self.b.set_value(theta_value[pos : pos+self.n_hidden], borrow=True)
                self.b_prime.set_value(theta_value[pos+self.n_hidden:], borrow=True)

        def Wb2theta():
            theta = self.W.get_value().flatten()
            if not self.use_tied_weight:
                theta = numpy.hstack((theta, self.W_prime.get_value().flatten()))
            if self.use_biases:
                theta = numpy.hstack((theta, self.b.get_value()))
                theta = numpy.hstack((theta, self.b_prime.get_value()))
            return theta

        def train_func(theta_value):
            theta2Wb(theta_value)
            return self.cost_fun()

        def train_fn_grad(theta_value):
            theta2Wb(theta_value)
            grads = [x.flatten() for x in self.cost_derive_fun()]
            return numpy.hstack(grads)

        def callback(theta_value):
            costs.append(train_func(theta_value))

        costs = []

        options = {
            'disp': show,
            'maxiter': max_iter,
            }
        opt_res = opt.minimize(
            train_func,
            method=opt_method,  # 'BFGS' 'L-BFGS-B' seems to only support float64
            x0=Wb2theta(),
            jac=train_fn_grad,
            callback=callback,
            options=options
        )

        return costs

    def pretrain_minibatch(self, max_iter=50, learn_rate=0.2, batch_size=20, show=False):

        self.learning_rate = learn_rate
        n_smp = self.x_train.get_value(borrow=True).shape[0]
        if batch_size < 1:
            batch_size = n_smp
        n_batches = n_smp/batch_size
        costs = []
        for epoch in range(max_iter):

            # generate an order to shuffle the rows of x
            shuffle_idx = self.numpy_rng.permutation(n_smp)
            # go through the training set
            c = []
            for b in xrange(n_batches):
                c.append(self.pretrainFun(
                    indices=shuffle_idx[b*batch_size:(b+1)*batch_size]))
            costs.append(numpy.mean(c))

            if show:
                print 'epoch %d, cost ' % epoch,
                print costs[-1]

        return costs

    def get_hidden_output(self, x):
        return self.hidden_output_fun(x)
