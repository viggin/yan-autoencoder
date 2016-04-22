"""
Yet ANother autoencoder toolbox.

An implementation of stacked autoencoder using the Theano library.

ref: http://deeplearning.net/tutorial/contents.html

Please cite: Ke Yan, and David Zhang, "Correcting Instrumental Variation
and Time-varying Drift: A Transfer Learning Approach with Autoencoders,"
accepted by Instrumentation and Measurement, IEEE Transactions on

Copyright 2016 YAN Ke, Tsinghua Univ. http://yanke23.com , xjed09@gmail.com
"""

import numpy
import theano

from yan_mlp import YanMlp
from yan_ae import YanAe


class YanSae(object):
    """Yet ANother stacked autoencoder, dependent on the YanAe class.

    Improved from http://deeplearning.net/tutorial/SdA.html

    The denoising, sparsity, L1/L2 regularization, and weight tying strategies can be set.

    """

    def __init__(
        self,
        n_visible=128,  # num of input neurons
        n_hidden=[6],  # list of hidden neurons, only the encoder part (the first
                       # hidden layer to the innermost hidden layer)
        act_fun_names=['lin'],  # list of activation functions of the encoder hidden layers,
                                # see func act_fun_from_name in yan_utils.py.
                                # Note: If use bounded func like tanh, you should make sure the
                                # scale of the data fit the output scale of the func
        corruption_levels=[0.3],  # list of corruption_levels of the encoder hidden layers
                                  # if >0, use the denoising strategy
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

        self.sub_aes = []
        self.params = []
        self.n_stacks = len(n_hidden)
        self.act_fun_names = act_fun_names
        self.corruption_levels = corruption_levels
        self.rng = (numpy.random.RandomState(6799) if rng is None else rng)

        assert self.n_stacks > 0
        if len(act_fun_names) == 1:
            act_fun_names *= self.n_stacks
        if len(corruption_levels) == 1:
            corruption_levels *= self.n_stacks

        self.n_units = [n_visible] + n_hidden
        for i in xrange(self.n_stacks):
            sub_ae = YanAe(
                n_visible=self.n_units[i],
                n_hidden=self.n_units[i+1],
                act_fun_name_vis=act_fun_names[i],
                act_fun_name_hid=act_fun_names[i],
                corruption_level=corruption_levels[i],
                sparsity_reg=sparsity_reg,
                sparsity_target=sparsity_target,
                L1_reg=L1_reg,
                L2_reg=L2_reg,
                use_tied_weight=use_tied_weight,
                use_biases=use_biases,
                loss_fun_name=loss_fun_name,
                rng=self.rng)
            self.sub_aes.append(sub_ae)

    def pretrain(self, x, max_iter=50, opt_method='CG',
                 learn_rate=0.2, batch_size=20,  # only for minibatch method
                 show=[False, False, False]  # [running msg, optimization msg, cost plot]
                 ):
        """Unsupervised pretrain each shallow AE"""

        self.x_train = theano.shared(value=x, borrow=True)
        x1 = x
        if show[0]:
            print(('pre-training the stacked autoencoder (%d stacks) with %d samples and '
                  '%d features..') % (self.n_stacks, x.shape[0], x.shape[1]))

        # Pretrain layer-wisely
        for i in xrange(self.n_stacks):
            self.sub_aes[i].pretrain(x1, max_iter, opt_method, learn_rate, batch_size, show)
            x1 = self.sub_aes[i].get_hidden_output(x1)

    def get_hidden_output(self, x):
        out = x
        for ae in self.sub_aes:
            out = ae.get_hidden_output(out)
        return out

    def fine_tune_sup(self, x, y, loss_fun_name,
                      max_iter=50, opt_method='CG',
                      L1_reg=0.00, L2_reg=0.00,
                      learn_rate=0.2, batch_size=20,  # only for minibatch method
                      show=[False, False, False]  # [running msg, optimization msg, cost plot]
                      ):
        """Supervisedly finetune the encoder part"""

        # construct an MLP with the encoder part
        self.mlp_sup = YanMlp(n_input=self.n_units[0], rng=self.rng)
        for i in range(self.n_stacks):
            ae = self.sub_aes[i]
            self.mlp_sup.add_layer(n_output=self.n_units[i+1],
                                   W=ae.W, b=ae.b,
                                   act_fun_name=self.act_fun_names[i])

        n_class = y.max() if y.ndim == 1 else y.shape[1]
        self.mlp_sup.add_layer(n_output=n_class, act_fun_name='softmax')
        self.mlp_sup.set_model(loss_fun_name=loss_fun_name, L1_reg=L1_reg,
                               L2_reg=L2_reg, )
        self.mlp_sup.train(x, y,
                           opt_method=opt_method,
                           learn_rate=learn_rate,
                           max_iter=max_iter,
                           batch_size=batch_size,
                           show=show)

    def fine_tune_unsup(self, x, loss_fun_name='mse',
                        max_iter=50, opt_method='CG',
                        L1_reg=0.00, L2_reg=0.001,
                        learn_rate=0.2, batch_size=20,  # only for minibatch method
                        show=[False, False, False]):
        """
        Unsupervisedly finetune the encoder part
        Will set use_tied_weight to False and change W_prime

        """
        # construct an MLP with the encoder part
        self.mlp_unsup = YanMlp(n_input=self.n_units[0], rng=self.rng)
        for i in range(self.n_stacks):
            ae = self.sub_aes[i]
            self.mlp_unsup.add_layer(n_output=self.n_units[i+1],
                                     W=ae.W, b=ae.b,
                                     act_fun_name=self.act_fun_names[i])

        for i in range(self.n_stacks):
            ae = self.sub_aes[-i-1]
            self.mlp_unsup.add_layer(n_output=self.n_units[-i-2],
                                     W=(None if ae.use_tied_weight else ae.W_prime),
                                     b=ae.b_prime,
                                     act_fun_name=self.act_fun_names[-i-1])
            ae.W_prime = self.mlp_unsup.layers[-1].W  # replace original W_prime
            ae.use_tied_weight = False

        self.mlp_unsup.set_model(loss_fun_name=loss_fun_name, L1_reg=L1_reg, L2_reg=L2_reg, )
        self.mlp_unsup.train(x, x,
                             opt_method=opt_method,
                             learn_rate=learn_rate,
                             max_iter=max_iter,
                             batch_size=batch_size,
                             show=show)

