"""
An implementation of drift correction autoencoder (DCAE)
ref: Ke Yan, and David Zhang, "Correcting Instrumental Variation
 and Time-varying Drift: A Transfer Learning Approach with Autoencoders,"
 accepted by Instrumentation and Measurement, IEEE Transactions on
Copyright 2016 YAN Ke, Tsinghua Univ. http://yanke23.com , xjed09@gmail.com
"""
import timeit
import os
import copy

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

import theano
import theano.tensor as tensor
from theano.tensor.shared_randomstreams import RandomStreams

from yan_mlp import YanMlp
from yan_ae import YanAe
from yan_utils import act_fun_from_name, loss_fun_from_name, make_one_hot_target, \
    opt_name_from_abbrv


class Dcae(object):
    """Drift correction autoencoder (DCAE) with correction layer.

    An autoencoder designed to correct instrumental variation and time-varying drift
    in sensor systems with the help of transfer samples.
    """

    FLOATX = theano.config.floatX

    def __init__(self,
                 sae_obj,  # an YanSae object trained before, used to initialize Dcae
                 n_domain_ft=1,  # number of domain features, see the ref
                 use_correction_layer=False,
                 rng=None):

        self.n_domain_ft = n_domain_ft

        # labeled data and their domain features
        self.x_labeled = tensor.matrix(name='x_labeled', dtype=self.FLOATX)
        self.xl_domain_ft = tensor.matrix(name='xl_domain_ft', dtype=self.FLOATX)

        self.n_units = [sae_obj.n_units[0]]  # stores the number of units in all layers,
                                             # doesn't consider the corrector part
        self.rng = (np.random.RandomState(1999) if rng is None else rng)
        self.n_stacks = sae_obj.n_stacks
        self.theano_rng = RandomStreams(self.rng.randint(2 ** 30))  # for denoising
        self.corruption_level = sae_obj.corruption_levels[0]
        self.use_correction_layer = use_correction_layer

        self.params = []  # all network weight matrices and bias vectors to be learned
        self.Ws = []  # weight matrices
        self.bs = []  # bias vectors
        self.act_funs = []
        for i in range(self.n_stacks):
            self.n_units += [sae_obj.n_units[i+1]]
            ae = sae_obj.sub_aes[i]
            self.Ws += [theano.shared(ae.W.get_value(), borrow=False, name=('W%d' % i))]
            self.bs += [theano.shared(ae.b.get_value(), borrow=False, name=('b%d' % i))]
            self.act_funs += [copy.deepcopy(ae.act_fun_hid)]
            self.params += [self.Ws[-1], self.bs[-1]]

        for i in range(self.n_stacks):
            self.n_units += [sae_obj.n_units[-i-2]]
            ae = sae_obj.sub_aes[-i-1]
            W_prime = ae.W.get_value().transpose()\
                if ae.use_tied_weight else ae.W_prime.get_value()
            self.Ws += [theano.shared(W_prime, borrow=False,
                                      name=('W\'%d' % (self.n_stacks-1-i)))]
            self.bs += [theano.shared(ae.b_prime.get_value(), borrow=False,
                                      name=('b\'%d' % (self.n_stacks-1-i)))]
            self.act_funs += [copy.deepcopy(ae.act_fun_hid)]

        # source transfer samples and their domain features
        self.t_src = tensor.matrix(name='t_src', dtype=self.FLOATX)
        self.ts_domain_ft = tensor.matrix(name='ts_domain_ft', dtype=self.FLOATX)

        # target transfer samples and their domain features
        self.t_tar = tensor.matrix(name='t_tar', dtype=self.FLOATX)
        self.tt_domain_ft = tensor.matrix(name='tt_domain_ft', dtype=self.FLOATX)

        # all data (don't care labeled or unlabeled) and their domain features
        self.x_all = tensor.matrix(name='x_all', dtype=self.FLOATX)
        self.xa_domain_ft = tensor.matrix(name='xa_domain_ft', dtype=self.FLOATX)

    def _change_correction_unit(self):
        """Add corrector part to the network"""

        if self.use_correction_layer:  # only DCAE with correction layer has Wd0
            initial_Wd0 = np.asarray(
                    self.rng.uniform(
                        low=-np.sqrt(6. / (self.n_domain_ft + self.n_cor_unit)),
                        high=np.sqrt(6. / (self.n_domain_ft + self.n_cor_unit)),
                        size=(self.n_domain_ft, self.n_cor_unit)
                    ),
                    dtype=self.FLOATX
                )*0  # init with 0 seems to be better?
            self.Wd0 = theano.shared(value=initial_Wd0,
                                     name='Wd0', borrow=True)
            self.params = [self.Wd0] + self.params

        initial_Wd1 = np.asarray(
            self.rng.uniform(
                low=-np.sqrt(6. / (self.n_cor_unit + self.n_units[1])),
                high=np.sqrt(6. / (self.n_cor_unit + self.n_units[1])),
                size=(self.n_cor_unit, self.n_units[1])
            ),
            dtype=self.FLOATX
        )
        self.Wd1 = theano.shared(value=initial_Wd1,  # can't init with zeros
                                 name='Wd1')
        self.params = [self.Wd1] + self.params

        self.xl_hid_out_tensor = self._compute_hidden_output_tensor(
            self.x_labeled, self.xl_domain_ft)
        self.ts_hid_out_tensor = \
            self._compute_hidden_output_tensor(self.t_src, self.ts_domain_ft)
        self.tt_hid_out_tensor = \
            self._compute_hidden_output_tensor(self.t_tar, self.tt_domain_ft)
        self.xa_rec_out_tensor = self._compute_reconstr_output_tensor(
            self.x_all, self.xa_domain_ft)

    def _compute_hidden_output_tensor(self, x, domain_ft):

        # output of the corrector part (which takes domain_ft as input)
        if self.use_correction_layer:  # only DCAE with correction layer has Wd0
            lin_output = tensor.dot(domain_ft, self.Wd0)
            cor_output = lin_output if self.cor_act_fun is None \
                else self.cor_act_fun(lin_output)  
        else:
            cor_output = domain_ft
       
        hid_val = x
        for i in range(self.n_stacks):
            # denoising strategy is not used
            lin_output = tensor.dot(hid_val, self.Ws[i]) + self.bs[i]

            # the output of the corrector part is added to the input of the
            # first hidden layer
            if i == 0:
                lin_output += tensor.dot(cor_output, self.Wd1)
            hid_val = lin_output if self.act_funs[i] is None \
                else self.act_funs[i](lin_output)

        return hid_val

    def _compute_reconstr_output_tensor(self, x, domain_ft):

        # output of the corrector part (which takes domain_ft as input)
        if self.use_correction_layer:  # only DCAE with correction layer has Wd0
            lin_output = tensor.dot(domain_ft, self.Wd0)
            cor_output = lin_output if self.cor_act_fun is None \
                else self.cor_act_fun(lin_output)
        else:
            cor_output = domain_ft

        hid_val = x
        for i in range(self.n_stacks*2):

            # the output of the corrector part is subtracted from the output of the
            # last hidden layer
            if i == self.n_stacks*2-1:
                hid_val -= tensor.dot(cor_output, self.Wd1)

            lin_output = tensor.dot(hid_val, self.Ws[i]) + self.bs[i]

            # the output of the corrector part is added to the input of the
            # first hidden layer
            if i == 0:
                lin_output += tensor.dot(cor_output, self.Wd1)

            hid_val = lin_output if self.act_funs[i] is None \
                else self.act_funs[i](lin_output)

        return hid_val

    def set_model(self, n_labels,
                  L1_reg=0.00,
                  L2_reg=0.001,

                  # the prediction layer is attached after the hidden output layer, see
                  # the ref and act_fun_from_name in yan_utils.py
                  pred_layer_act_fun_name='softmax',

                  sup_loss_fun_name='ce',  # supervised loss, ce for classification, mse for regression
                                           # see loss_fun_from_name in yan_utils.py
                  sup_reg=1,  # weight parameter for the supervised loss, lambda_1 in the ref
                  trans_reg=1,  # weight parameter for the transfer loss, lambda_2 in the ref

                  # these two args are useful only if use_correction_layer = True
                  n_cor_unit=5,  # number of units in the correction layer, h_cor in the ref
                  cor_act_fun_name='tanh',  # act fun of the correction layer, sigma_cor in the ref

                  # weight parameter for the reconstr error of all data, actually should be
                  # 1 in the ref since it is the first term of the objective function, but
                  # we add this arg to increase the flexibility of the function. Setting
                  # reconstr_reg=2 is equivalent to setting sup_reg/=2; trans_reg/=2
                  reconstr_reg=1.,
                  ):
        """Set the parameters of the DCAE"""

        self.L1_reg = L1_reg
        self.L2_reg = L2_reg
        self.reconstr_reg = reconstr_reg
        self.pred_layer_act_fun_name = pred_layer_act_fun_name
        self.n_labels = n_labels
        self.sup_loss_fun_name = sup_loss_fun_name
        self.trans_reg = trans_reg
        self.sup_reg = sup_reg

        self.trans_loss_fun = loss_fun_from_name("mse")
        self.n_domain_ft = self.n_domain_ft
        self.n_cor_unit = n_cor_unit if self.use_correction_layer else self.n_domain_ft
        self.cor_act_fun = act_fun_from_name(cor_act_fun_name)

        self._change_correction_unit()

        if self.reconstr_reg > 0:
            for i in range(self.n_stacks,self.n_stacks*2):
                self.params += [self.Ws[i], self.bs[i]]

        if self.sup_reg > 0:
            self.pred_act_fun = act_fun_from_name(pred_layer_act_fun_name)
            init_W = np.asarray(
                    self.rng.uniform(
                        low=-np.sqrt(6. / (self.n_units[self.n_stacks] + n_labels)),
                        high=np.sqrt(6. / (self.n_units[self.n_stacks] + n_labels)),
                        size=(self.n_units[self.n_stacks], n_labels)
                    ),
                    dtype=self.FLOATX
                )
            if self.pred_act_fun == tensor.nnet.sigmoid:
                init_W *= 4
            self.Ws += [theano.shared(value=init_W, name='W_pred', borrow=True)]

            init_b = np.zeros((n_labels,), dtype=self.FLOATX)
            self.bs += [theano.shared(value=init_b, name='b_pred', borrow=True)]
            self.params += [self.Ws[-1], self.bs[-1]]
            self.y_labeled = tensor.matrix(name='y_labeled', dtype=self.FLOATX)
            self.sup_loss_fun = loss_fun_from_name(sup_loss_fun_name)

    def _compute_tensors_for_train(self):
        self.cost_tensor = 0

        # supervised cost
        if self.sup_reg > 0:
            pred_tensor = tensor.dot(self.xl_hid_out_tensor, self.Ws[-1]) + self.bs[-1]
            if self.pred_act_fun is not None:
                pred_tensor = self.pred_act_fun(pred_tensor)

            loss_tensor = self.sup_loss_fun(self.y_labeled, pred_tensor)
            self.cost_tensor += tensor.mean(loss_tensor) * self.sup_reg

        # reconstruction error of all data
        if self.reconstr_reg > 0:
            self.xa_rec_loss_tensor = \
                self.trans_loss_fun(self.xa_rec_out_tensor, self.x_all)
            self.cost_tensor += tensor.mean(self.xa_rec_loss_tensor) * self.reconstr_reg

        # regularization of W
        if self.L1_reg > 0:
            L1 = 0
            for W in self.Ws:
                L1 += abs(W).sum()
            self.cost_tensor += L1*self.L1_reg
        if self.L2_reg > 0:
            L2 = 0
            for W in self.Ws:
                L2 += (W**2).sum()
            self.cost_tensor += L2*self.L2_reg

        # transfer error, aligning the hidden representation of the transfer samples
        if self.trans_reg > 0:
            self.trLoss_tensor = self.trans_loss_fun(
                self.ts_hid_out_tensor, self.tt_hid_out_tensor)
            self.cost_tensor += tensor.mean(self.trans_smp_weights * self.trLoss_tensor) * self.trans_reg

    def _compute_funs_for_train(self, is_minibatch, learn_rate=.2):

        self._compute_tensors_for_train()
        givens = {}
        if self.sup_reg > 0:  # has supervised loss
            givens[self.x_labeled] = self.x_labeled_train
            givens[self.xl_domain_ft] = self.xl_domain_ft_train
            givens[self.y_labeled] = self.y_labeled_train
        if self.reconstr_reg > 0:  # has unsupervised loss
            givens[self.x_all] = self.x_all_train
            givens[self.xa_domain_ft] = self.xa_domain_ft_train
        if self.trans_reg > 0:  # has transfer sample cost
            givens[self.t_src] = self.t_src_train
            givens[self.ts_domain_ft] = self.ts_domain_ft_train
            givens[self.t_tar] = self.t_tar_train
            givens[self.tt_domain_ft] = self.tt_domain_ft_train

        if is_minibatch:
            indices = tensor.lvector('indices')  # indices of samples to a minibatch
            self.cost_derive_tensor = tensor.grad(self.cost_tensor, self.params)
            updates = [
                (param, param - learn_rate * gparam)
                for param, gparam in zip(self.params, self.cost_derive_tensor)
            ]
            if hasattr(self, 'y'):  # has supervised loss
                givens[self.x_labeled] = self.x_labeled_train[indices]
                givens[self.xl_domain_ft] = self.xl_domain_ft_train[indices]
                givens[self.y_labeled] = self.y_labeled_train[indices]
                self.minibatch_trainFun = theano.function(
                    inputs=[indices],
                    outputs=self.cost_tensor,
                    updates=updates,
                    givens=givens
                )

        else:  # use scipy optimization
            self.cost_fun = theano.function(
                    inputs=[],
                    outputs=self.cost_tensor,
                    givens=givens
                )

            self.cost_derive_tensor = tensor.grad(self.cost_tensor, self.params)
            self.cost_derive_fun = theano.function(
                        inputs=[],
                        outputs=self.cost_derive_tensor,
                        givens=givens
                    )

    def _settle_transfer_samples(self, trans_smp_tuple, tr_domain_ft_tuple):

        n_trans_smp = trans_smp_tuple[0].shape[0]
        self.t_src_train = theano.shared(name='t_src_train',
                                         value=trans_smp_tuple[0], borrow=False)
        self.t_tar_train = theano.shared(name='t_tar_train',
                                         value=trans_smp_tuple[1], borrow=False)
        tr_domain_ft_tuple1 = [copy.deepcopy(tr_domain_ft_tuple[0]),
                               copy.deepcopy(tr_domain_ft_tuple[1])]

        if type(tr_domain_ft_tuple[0]) != np.ndarray: # in case of numbers as domain ft
            tr_domain_ft_tuple1[0] = np.tile(tr_domain_ft_tuple[0],
                                             (n_trans_smp, 1)).astype(self.FLOATX)
        if type(tr_domain_ft_tuple[1]) != np.ndarray:
            tr_domain_ft_tuple1[1] = np.tile(tr_domain_ft_tuple[1],
                                             (n_trans_smp, 1)).astype(self.FLOATX)

        self.ts_domain_ft_train = theano.shared(
            name='ts_domain_ft_train', value=tr_domain_ft_tuple1[0], borrow=False)
        self.tt_domain_ft_train = theano.shared(
            name='tt_domain_ft_train', value=tr_domain_ft_tuple1[1], borrow=False)

    def train(self,
              # labeled data and their labels and domain features
              x_labeled, y_labeled,
              xl_domain_ft,  # array

              # all data (don't care labeled or unlabeled) and their domain features
              x_all, xa_domain_ft,


              trans_smp_tuple,  # 2-element tuple (src and target transfer samples)
                                # each element is an array
              tr_domain_ft_tuple,  # 2-element tuple (domain ft for src and target transfer
                                   #  samples), each element is an scalar or array

              max_iter=50, opt_method='CG',
              learn_rate=0.2, batch_size=20,  # only for minibatch method
              show=[False, False, False]  # [running msg, optimization msg, cost plot]
              ):
        """Optimize the overall objective function of DCAE"""

        self._settle_transfer_samples(trans_smp_tuple, tr_domain_ft_tuple)
        self.trans_smp_weights = np.ones((trans_smp_tuple[0].shape[0]))

        if y_labeled.ndim == 1 and self.sup_loss_fun_name == 'ce':
            y_labeled = make_one_hot_target(y_labeled)
        else:
            y_labeled = y_labeled.astype(self.FLOATX)  # copy
        opt_method = opt_name_from_abbrv(opt_method)
        n_smp = x_labeled.shape[0]

        if show[0]:
            print 'compiling the drift correction autoencoder..'
        self.x_labeled_train = theano.shared(name='x_train', value=x_labeled, borrow=True)
        self.x_all_train = theano.shared(name='x_all_train', value=x_all, borrow=True)
        self.y_labeled_train = theano.shared(name='y_train', value=y_labeled, borrow=True)

        # if type(xl_domain_ft) != np.ndarray:
        #     xl_domain_ft = np.tile(xl_domain_ft, (n_smp, 1)).astype(self.FLOATX)
        self.xl_domain_ft_train = theano.shared(name='xl_domain_ft_train',
                                                value=xl_domain_ft, borrow=True)
        self.xa_domain_ft_train = theano.shared(name='xa_domain_ft_train',
                                                value=xa_domain_ft, borrow=True)

        self._compute_funs_for_train(is_minibatch=(opt_method == 'minibatch'),
                                     learn_rate=learn_rate)

        if show[0]:
            print ("Optimizing using %s.." % opt_method)
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
        self.opt_costs = costs

        self.new_transfer_losses = self.get_transfer_losses()
        if show[0]:
            print 'transfer loss is %.4e' % self.new_transfer_losses.mean()
            print 'reconstruction loss is %.4e' % self.get_reconstr_loss().mean()

        self.hid_output_fun = theano.function(
                inputs=[self.x_labeled, self.xl_domain_ft],
                outputs=self.xl_hid_out_tensor,
            )

    def train_scipy(self, max_iter=50, opt_method='CG', show=False):

        def theta2Wb(theta_value):
            # watch the order of the params when extracting them from theta
            pos = 0
            self.Wd1.set_value(theta_value[pos:pos + self.n_cor_unit * self.n_units[1]].
                               reshape((self.n_cor_unit, self.n_units[1])), borrow=False)
            pos += self.n_cor_unit * self.n_units[1]
            if self.use_correction_layer:
                self.Wd0.set_value(theta_value[pos:pos+self.n_domain_ft * self.n_cor_unit].
                                   reshape((self.n_domain_ft, self.n_cor_unit)), borrow=False)
                pos += self.n_domain_ft * self.n_cor_unit

            for W, b in zip(self.Ws, self.bs):
                W_len = W.get_value(borrow=True).size
                b_len = b.get_value(borrow=True).size
                W.set_value(theta_value[pos:pos+W_len].
                            reshape(W.get_value(borrow=True).shape), borrow=False)
                b.set_value(theta_value[pos+W_len : pos+W_len+b_len], borrow=False)
                pos += (W_len+b_len)

        def Wb2theta():
            # watch the order of the params when concatenating
            theta = self.Wd1.get_value().flatten()
            if self.use_correction_layer:
                theta = np.hstack((theta, self.Wd0.get_value().flatten()))

            for W, b in zip(self.Ws, self.bs):
                theta = np.hstack((theta, W.get_value().flatten()))
                theta = np.hstack((theta, b.get_value()))
            return theta

        def train_fn(theta_value):
            theta2Wb(theta_value)
            return self.cost_fun()

        def train_fn_grad(theta_value):
            theta2Wb(theta_value)
            grads = [x.flatten() for x in self.cost_derive_fun()]
            return np.hstack(grads)

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
        n_smp = self.x_labeled_train.get_value(borrow=True).shape[0]
        if batch_size < 1:
            batch_size = n_smp
        n_batches = n_smp/batch_size
        costs = []
        for epoch in range(max_iter):

            # generate an order to shuffle the rows of x
            shuffle_idx = self.rng.permutation(n_smp)

            # go through the training set
            c = []
            for b in range(n_batches):
                c.append(self.minibatch_trainFun(
                    indices=shuffle_idx[b*batch_size:(b+1)*batch_size]))
            costs.append(np.mean(c))

            if show:
                print 'epoch %d, cost ' % (epoch),
                print costs[-1]

        return costs

    def get_hidden_output(self, x, domain_ft):
        if type(domain_ft) != np.ndarray:
            domain_ft = np.tile(domain_ft,(x.shape[0], 1)).astype(self.FLOATX)
        return self.hid_output_fun(x, domain_ft)

    def get_transfer_losses(self):
        if self.trans_reg == 0:
            return np.array([0])
        givens = {}
        givens[self.t_src] = self.t_src_train
        givens[self.ts_domain_ft] = self.ts_domain_ft_train
        givens[self.t_tar] = self.t_tar_train
        givens[self.tt_domain_ft] = self.tt_domain_ft_train
        trans_loss_value_fun = theano.function(
                inputs=[],
                outputs=self.trLoss_tensor,
                givens=givens
            )
        return trans_loss_value_fun()

    def get_reconstr_loss(self):
        if self.reconstr_reg == 0:
            return [0]
        givens = {}
        givens[self.x_all] = self.x_all_train
        givens[self.xa_domain_ft] = self.xa_domain_ft_train
        reconstr_loss_value_fun = theano.function(
                inputs=[],
                outputs=self.xa_rec_loss_tensor,
                givens=givens
            )
        return reconstr_loss_value_fun()
