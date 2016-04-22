"""
Yet Another Neural network toolbox.

An implementation of Marginalized Denoising Autoencoder using numpy.

ref: M. Chen, Z. Xu, K. Weinberger, and F. Sha, "Marginalized denoising autoencoders
  domain adaptation," arXiv preprint arXiv:1206.4683, 2012.

Copyright 2016 YAN Ke, Tsinghua Univ. http://yanke23.com , xjed09@gmail.com

"""

import numpy as np
from yan_utils import act_fun_from_name


class YanMsda(object):
    """Marginalized Denoising Autoencoder"""

    def __init__(
        self,
        n_layers=1,
        corrupt=.1,  # denoising strategy
        act_fun_name='tanh',
    ):

        self.n_layers = n_layers
        self.corrupt = corrupt
        self.Ws = []
        acts = {'tanh': np.tanh,
                'lin': None}
        self.activation = acts[act_fun_name]

    def train(self, x):
        n_smp = x.shape[0]
        x_current = x.transpose()
        for k in range(self.n_layers):
            x_current = np.vstack((x_current, np.ones((1, n_smp))))
            d = x_current.shape[0]
            q = np.vstack((np.ones((d-1, 1))*(1-self.corrupt), 1))
            S = np.dot(x_current, x_current.transpose())
            Q = S*(np.dot(q,q.transpose()))
            Q[range(d), range(d)] = q.flatten()*np.diag(S)
            P = S*np.tile(q.transpose(), (d, 1))
            W = np.dot(P[:-1, :], np.linalg.inv(Q + 1e-5*np.eye(d)))
            self.Ws += [W]
            x_current = np.tanh(np.dot(W, x_current))

    def get_hidden_output(self, x, id_layer=None):
        out = x.transpose()
        for k in range(self.n_layers):
            if id_layer is not None and k == id_layer:
                break
            out = np.vstack((out, np.ones((1, out.shape[1]))))
            out = np.dot(self.Ws[k], out)
            if self.activation is not None:
                out = self.activation(out)
        return out.transpose()
