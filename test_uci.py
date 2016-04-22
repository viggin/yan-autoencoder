"""
Test DCAE (as well as YanSae) in a dataset with time-varying drift.

The UCI dataset contains 10 batches of e-nose samples. We train on batch 1 and test on
batches 2-10, so drift correction needs to be performed by DCAE.

We have achieved the highest average accuracy in this dataset under this config, with
only 10 transfer samples needed in each batch.

See the ref: Ke Yan, and David Zhang, "Correcting Instrumental Variation
and Time-varying Drift: A Transfer Learning Approach with Autoencoders, "
accepted by Instrumentation and Measurement, IEEE Transactions on

Copyright 2016 YAN Ke, Tsinghua Univ. http://yanke23.com , xjed09@gmail.com

"""

import numpy as np
import timeit
import copy
import theano as tensor
from sklearn import preprocessing
from sklearn import linear_model, decomposition
from sklearn.svm import SVC
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat

from yan_sae import YanSae
from yan_ae import YanAe
from yan_utils import save2txt


def load_data():
    """
    Load from matlab .mat file
    :return data: a list of samples in 10 batches
    :return label: a list of labels in 10 batches (6 kinds of gases)
    :return tr_smp: pre-selected transfer samples in each pair (batch 1 as source,
        batch b as target)

    """
    data_dict = loadmat(r'UciData.mat')
    data0 = data_dict['data'][0].tolist()
    label0 = data_dict['label'][0].tolist()
    label = [np.int32(t.flatten()) for t in label0]

    # transfer samples pre-selected using the LLR algorithm
    t_src = data_dict['Tsrc'][0].tolist()
    t_tar = data_dict['Ttar'][0].tolist()

    # preprocess
    data_divider = 2
    data = []
    scaler0 = preprocessing.StandardScaler().fit(data0[0])
    for b in range(len(label)):
        data1 = scaler0.transform(data0[b])
        data1 /= data_divider
        data1 = data1.astype(tensor.config.floatX)
        data.append(data1)

    tr_smp = []
    for b in range(len(t_src)):
        data1 = scaler0.transform(t_src[b])
        data1 = data1[:n_trans_smp]/data_divider
        data1 = data1.astype(tensor.config.floatX)
        data2 = scaler0.transform(t_tar[b])
        data2 = data2[:n_trans_smp]/data_divider
        data2 = data2.astype(tensor.config.floatX)

        tr_smp.append((data1, data2))

    return data, label, tr_smp


def test():
    """test DCAE"""

    from dcae import Dcae
    (data, labels, tr_smp) = load_data()

    # pretrain and fine tune an SAE using only batch 1
    x_labeled = data[0]  # shallow copy
    y = labels[0]
    opt = 'cg'
    show = [True, True, True]
    max_iter = 1000

    sae = YanSae(n_visible=x_labeled.shape[1], n_hidden=n_hidden, 
                 act_fun_names=['tanh'], 
                 corruption_levels=[corrupt], 
                 sparsity_reg=.0,  # do not use sparsity
                 sparsity_target=.1, 
                 L1_reg=.0, 
                 L2_reg=.0, 
                 use_tied_weight=True, use_biases=True, rng=rng)
    sae.pretrain(x=x_labeled, max_iter=max_iter, opt_method=opt, show=show)
    sae.fine_tune_sup(x=x_labeled, y=y, loss_fun_name='ce', 
                      L1_reg=.0, L2_reg=0.001, 
                      max_iter=max_iter, opt_method=opt, show=show)

    # initialize a classification model
    clsf_model = linear_model.LogisticRegression(solver='liblinear', multi_class='ovr')
    # clsf_model = SVC(kernel='linear', C=100, gamma=.01)
    # clsf_model = RandomForestClassifier()
    # ft1 = sae.get_hidden_output(data[0])
    # clsf_model.fit(ft1, labels[0])
    # acc = [clsf_model.score(ft1, labels[0])]
    # print ' '.join('{:.4f}'.format(k) for k in acc)
    acc = []

    # domain ft on batch 1
    domain_ft_1 = [1., 0.]
    tr_smp_src = np.zeros((0, x_labeled.shape[1]))
    tr_smp_tars = np.zeros((0, x_labeled.shape[1]))
    ts_domain_ft = np.zeros((0, 2))
    tt_domain_fts = np.zeros((0, 2))
    x_all = data[0]  # here we found that reconstructing only batch 1 is better than
                     # reconstructing all data
    xa_domain_ft = np.tile(domain_ft_1, (data[0].shape[0], 1))

    # iterate on batches 2-10. For each batch b, train a DCAE based on labeled smp in
    # batch 1, and transfer sample pairs on batches 1 to b
    for b in range(1, len(labels)):

        dcae = Dcae(sae, n_domain_ft=2, use_correction_layer=use_correction_layer, rng=rng)
        dcae.set_model(n_labels=6,
                       L1_reg=0.,
                       L2_reg=0.,
                       trans_reg=trans_reg,
                       sup_reg=sup_reg,
                       n_cor_unit=n_cor_unit,
                       cor_act_fun_name='tanh',
                       reconstr_reg=1)

        # add transfer sample pairs on batch 1 and b to tr_smp_src and tr_smp_tars
        domain_ft_b = [1., b]
        n_trans_smp_real = tr_smp[b-1][0].shape[0]
        tr_smp_src = np.vstack((tr_smp_src, tr_smp[b-1][0]))
        tr_smp_tars = np.vstack((tr_smp_tars, tr_smp[b-1][1]))

        # add domain ft of transfer sample pairs to ts_domain_ft and tt_domain_fts
        ts_domain_ft = np.vstack((ts_domain_ft, np.tile(domain_ft_1, (n_trans_smp_real, 1))))
        tt_domain_fts = np.vstack((tt_domain_fts, np.tile(domain_ft_b, (n_trans_smp_real, 1))))
        trans_smp_1 = (tr_smp_src, tr_smp_tars)
        tr_domain_ft_1 = (ts_domain_ft, tt_domain_fts)

        dcae.train(x_labeled, y, xa_domain_ft,
                   x_all=x_all, xa_domain_ft=xa_domain_ft, 
                   trans_smp_tuple=trans_smp_1, tr_domain_ft_tuple=tr_domain_ft_1,
                   max_iter=max_iter, opt_method=opt, show=show)
        # print dcae.Wd1.get_value(), dcae.Wd0.get_value(), # dcae.bcor.get_value()
        # print dcae.Ws[0].get_value()
        # print dcae.get_transfer_losses()
        # trans_losses += [dcae.get_transfer_losses()]

        # get learned ft of batch 1 and b, train on batch 1 and test on b
        ft_labeled = dcae.get_hidden_output(data[0], domain_ft=domain_ft_1)
        clsf_model.fit(ft_labeled, labels[0])
        ft_test = dcae.get_hidden_output(data[b], domain_ft=domain_ft_b)
        acc.append(clsf_model.score(ft_test, labels[b]))
        print ' '.join('{:.4f}'.format(k) for k in acc)

    # show all acc
    acc.append(np.mean(acc))  # the last number is the avg
    print ' '.join('{:.4f}'.format(k) for k in acc)
    return acc


if __name__ == '__main__':

    n_trans_smp = 10
    corrupt = .1
    n_cor_unit = 6
    n_hidden = [30, 20]
    trans_reg = 2**-3
    sup_reg = 2**-4
    use_correction_layer = True
    rng = np.random.RandomState(0)

    test()
