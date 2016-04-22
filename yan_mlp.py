"""
Yet Another Neural network toolbox.

An implementation of multi-layer perceptron (fully connected network)
    using the Theano library.

ref: http://deeplearning.net/tutorial/contents.html

Please cite: Ke Yan, and David Zhang, "Correcting Instrumental Variation
and Time-varying Drift: A Transfer Learning Approach with Autoencoders,"
accepted by Instrumentation and Measurement, IEEE Transactions on

Copyright 2016 YAN Ke, Tsinghua Univ. http://yanke23.com , xjed09@gmail.com

"""
import os
import sys
import timeit
import copy

import numpy
import scipy.optimize as opt
import matplotlib.pyplot as plt

import theano
import theano.tensor as T

from yan_utils import act_fun_from_name, loss_fun_from_name, make_one_hot_target, opt_name_from_abbrv


class YanLayer(object):
    """A layer in the MLP"""

    def __init__(self, x,
                 n_in, n_out,  # number of input and output of the layer

                 # initial weight matrix and bias vector
                 # If None, will be randomly initialized
                 # If not None, should be shared var (will be shared with other nets)
                 # The sharing strategy will be used when finetuning an SAE
                 W=None, b=None,

                 act_fun_name='sigm', rng=None):

        rng = (numpy.random.RandomState(12**3) if rng is None else rng)
        self.activation = act_fun_from_name(act_fun_name)
        if W is None:
            initial_W = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if self.activation == T.nnet.sigmoid:
                initial_W *= 4  # according to http://deeplearning.net/tutorial/contents.html
            W = theano.shared(value=initial_W, name='W', borrow=True)
        self.W = W

        if b is None:
            initial_b = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=initial_b, name='b', borrow=True)
        self.b = b

        lin_output = T.dot(x, self.W) + self.b
        self.output = (
            lin_output if self.activation is None
            else self.activation(lin_output)
        )
        self.params = [self.W, self.b]


class YanMlp(object):
    """Yet ANother multi-layer perceptron (fully connected network).

    Improved from http://deeplearning.net/tutorial/mlp.html
    The denoising, sparsity, L1/L2 regularization, and weight tying strategies can be set.

    """

    def __init__(self, n_input, rng=None):
        self.n_units = [n_input]
        self.n_layers = 0
        self.rng = (numpy.random.RandomState(714285) if rng is None else rng)
        self.layers = []
        self.x = T.matrix(name='x', dtype=theano.config.floatX)
        self.params = []

    def add_layer(self, n_output, W=None, b=None,
                  act_fun_name='sigm'  # see func act_fun_from_name in yan_utils.py
                  ):
        """Add one layer after the current MLP.
        W and b are the initial weight matrix and bias vector.
        If set to None, they will be randomly initialized.
        If not None, they should be shared variables in Theano (will be shared with other nets).
        The sharing strategy will be used when finetuning an SAE

        """
        if self.n_layers == 0:
            input_tensor = self.x
        else:
            input_tensor = self.layers[-1].output
        self.layers.append(YanLayer(
            x=input_tensor,
            rng=self.rng,
            n_in=self.n_units[-1],
            n_out=n_output,
            act_fun_name=act_fun_name,
            W=W, b=b,
        ))
        self.n_layers += 1
        self.n_units.append(n_output)
        self.params.extend([self.layers[-1].W, self.layers[-1].b])

    def set_model(self, L1_reg=0.00, L2_reg=0.001,
                  loss_fun_name='ce',  # ce for classification, mse for regression
                                       # see loss_fun_from_name in yan_utils.py
                  ):
        """Set the parameters of the MLP"""

        self.y = T.matrix(name='y', dtype=theano.config.floatX)
        loss_fun = loss_fun_from_name(loss_fun_name)
        pred_tensor = self.layers[-1].output
        loss_tensor = loss_fun(self.y, pred_tensor)

        L1 = 0
        L2 = 0
        for layer in self.layers:
            L1 += abs(layer.W).sum()
            L2 += (layer.W**2).sum()

        self.cost_tensor = T.mean(loss_tensor) + L1*L1_reg + L2*L2_reg
        self.cost_derive_tensor = T.grad(self.cost_tensor, self.params)  # gradients
        self.output_fun = theano.function(
                inputs=[self.x],
                outputs=pred_tensor,
            )

    def _compute_funs_for_train(self, is_minibatch, learn_rate=.2):

        if is_minibatch:
            indices = T.lvector('indices')  # indices of samples to a minibatch
            updates = [
                (param, param - learn_rate * gparam)
                for param, gparam in zip(self.params, self.cost_derive_tensor)
            ]

            self.minibatch_trainFun = theano.function(
                inputs=[indices],
                outputs=self.cost_tensor,
                updates=updates,
                givens={
                    self.x: self.x_train[indices],
                    self.y: self.y_train[indices]
                }
            )

        else:  # use scipy optimization
            self.cost_fun = theano.function(
                    inputs=[],
                    outputs=self.cost_tensor,
                    givens={
                        self.x:  self.x_train,
                        self.y:  self.y_train
                    }
                )

            # compute the gradients of the cost of the MLP with respect
            # to its parameters
            self.cost_derive_fun = theano.function(
                    inputs=[],
                    outputs=self.cost_derive_tensor,
                    givens={
                        self.x:  self.x_train,
                        self.y:  self.y_train
                    }
                )

    def train(self, x, y,
              max_iter=50, opt_method='CG',
              learn_rate=0.2, batch_size=20,  # only for minibatch method
              show=[False, False, False]  # [running msg, optimization msg, cost plot]
              ):

        if self.n_units[-1] > 1 and y.ndim == 1:
            y = make_one_hot_target(y)
        else:
            y = y.astype(theano.config.floatX)  # copy
        opt_method = opt_name_from_abbrv(opt_method)

        if show[0]:
            print 'compiling the multi-layer perceptron..'
        self.x_train = theano.shared(name='x', value=x, borrow=True)
        self.y_train = theano.shared(name='y', value=y, borrow=True)
        self._compute_funs_for_train(is_minibatch=(opt_method == 'minibatch'),
                                     learn_rate=learn_rate)
        if show[0]:
            print(('training the MLP (%d layers) with %d samples and '
                   '%d features using %s..') % (self.n_layers, x.shape[0], x.shape[1], opt_method))
        start_time = timeit.default_timer()

        if opt_method == 'minibatch':
            costs = self.train_minibatch(max_iter, learn_rate, batch_size, show[1])
        else:
            costs = self.train_scipy(max_iter, opt_method, show[1])

        end_time = timeit.default_timer()

        if show[0]:
            print 'The optimization for file ' + os.path.split(__file__)[1] + \
              ' ran for %.1fs' % (end_time - start_time)
        if show[2]:
            plt.figure('cost per epoch')
            plt.plot(range(len(costs)), costs)
            # plt.show()

    def train_scipy(self, max_iter=50, opt_method='CG', show=False):

        def theta2Wb(theta_value):
                pos = 0
                for i in range(self.n_layers):
                    layer = self.layers[i]
                    W_len = self.n_units[i]*self.n_units[i+1]
                    layer.W.set_value(theta_value[pos : pos+W_len].
                                      reshape((self.n_units[i], self.n_units[i+1])),
                                      borrow=False)
                    layer.b.set_value(theta_value[pos+W_len : pos+W_len+self.n_units[i+1]],
                                      borrow=False)
                    pos += W_len+self.n_units[i+1]

        def Wb2theta():
            theta = numpy.array([])
            for i in range(self.n_layers):
                theta = numpy.hstack((theta, self.layers[i].W.get_value().flatten()))
                theta = numpy.hstack((theta, self.layers[i].b.get_value()))
            return theta

        def train_fn(theta_value):
            theta2Wb(theta_value)
            return self.cost_fun()

        def train_fn_grad(theta_value):
            theta2Wb(theta_value)
            grads = [x.flatten() for x in self.cost_derive_fun()]
            return numpy.hstack(grads)

        def callback(theta_value):
            costs.append(train_fn(theta_value))

        costs = []

        options = {
            'disp': show,
            'maxiter': max_iter,
            }
        opt_res = opt.minimize(
            train_fn,
            method=opt_method,  # 'BFGS' 'L-BFGS-B' seems to only support float64
            x0=Wb2theta(),
            jac=train_fn_grad,
            callback=callback,
            options=options
        )

        return costs

    def train_minibatch(self, max_iter=50, learn_rate=0.2, batch_size=20
                        , show=False):

        self.learning_rate = learn_rate
        n_smp = self.x_train.get_value(borrow=True).shape[0]
        if batch_size < 1:
            batch_size = n_smp
        n_batches = n_smp/batch_size
        costs = []
        for epoch in range(max_iter):

            # generate an order to shuffle the rows of x
            shuffle_idx = self.rng.permutation(n_smp)
            # train_set_x.set_value(train_set_x.get_value()[shuffle_idx])
            # go through the training set
            c = []
            for b in range(n_batches):
                c.append(self.minibatch_trainFun(
                    indices=shuffle_idx[b*batch_size : (b+1)*batch_size]))
            costs.append(numpy.mean(c))

            if show:
                print 'epoch %d, cost ' % (epoch),
                print costs[-1]

        return costs

    def get_output(self, x):
        return self.output_fun(x)
