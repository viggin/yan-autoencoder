"""
Test DCAE (as well as YanSae) in a dataset with instrumental variation.

The corn dataset was collected by three devices. We train on one device and test on
the other two, so calibration transfer needs to be performed by DCAE.

See the ref: Ke Yan, and David Zhang, "Correcting Instrumental Variation
and Time-varying Drift: A Transfer Learning Approach with Autoencoders,"
accepted by Instrumentation and Measurement, IEEE Transactions on

Copyright 2016 YAN Ke, Tsinghua Univ. http://yanke23.com , xjed09@gmail.com

"""

from scipy.io import loadmat
import numpy as np
import timeit
import copy
import theano
from sklearn import preprocessing
from sklearn import linear_model, decomposition
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat

from yan_sae import YanSae
from yan_ae import YanAe
from yan_utils import save2txt


def load_data():
    """
    Load from matlab .mat file
    :return ft_all: a list of samples in 3 devices
    :return y: an array of 4 regression labels in 80 samples
    :return trans_smp_list: indices of pre-selected transfer samples in each cross-validation
    :return scaler_y: y is z-scored in this function, we will use it to compute the real y

    """
    print 'loading data...'
    data_dict = loadmat(r'cornData.mat')
    for varName in ['Y', 'ftAll', 'trList']:
        exec("%s=data_dict['%s']" % (varName, varName))
    ft_all = ftAll[0]
    id_master = 0

    # preprocess
    ft_prep = np.vstack(ft_all)
    scaler0 = preprocessing.StandardScaler().fit(ft_prep)

    for i in range(len(ft_all)):
        ft_all[i] = scaler0.transform(ft_all[i])
        ft_all[i] = ft_all[i].astype(theano.config.floatX)/2
        # plt.figure('data')
        # plt.plot(ftAll[i].T)
    # plt.show()

    trans_smp_list = trList-1  # from matlab's 1-based indices to python's 0-based ones

    scaler_y = preprocessing.StandardScaler().fit(Y)  # normalize Y
    y = scaler_y.transform(Y)
    return ft_all, y, trans_smp_list, scaler_y


def run_reg(ft_train, ft_test, y_train, mask_test, preds):
    """Train regression models and fit ft_test and save to preds"""

    n_exp = y_train.shape[1]
    n_devices = len(ft_test)
    err = np.empty((n_devices, n_exp))
    reg_model = linear_model.Ridge(alpha=1, fit_intercept=True)
    # reg_model = SVR(kernel='rbf', C=1e5, gamma=.2)

    for i_exp in range(n_exp):
        reg_model.fit(ft_train, y_train[:, i_exp])
        for i_dev in range(n_devices):
            preds[i_dev][mask_test, i_exp] = reg_model.predict(ft_test[i_dev])


def test():
    """Test DCAE"""

    from dcae import Dcae
    ft_all, y, trans_smp_list, scaler_y = load_data()
    n_exp = y.shape[1]
    n_devices = len(ft_all)
    n_cross_val = 4
    n_smp = y.shape[0]
    n_ft = ft_all[0].shape[1]
    err = np.empty((n_devices-1, n_exp))
    id_master = 0
    id_slave = [1, 2]

    # pretrain a stacked AE
    data_pretr = np.vstack(ft_all)  # use all data for pretrain
    opt = 'cg'
    show = [True, True, False]
    max_iter = 1000
    learn_rate = .1
    batch_size = 20

    sae = YanSae(n_visible=data_pretr.shape[1],
                 n_hidden=n_hidden,
                 act_fun_names=['lin'],
                 corruption_levels=[corrupt],
                 sparsity_reg=.0,  # do not use sparsity
                 sparsity_target=.1,
                 L1_reg=.0,
                 L2_reg=.001,
                 use_tied_weight=True,
                 use_biases=True,
                 rng=rng)
    sae.pretrain(x=data_pretr,
                 max_iter=max_iter,
                 opt_method=opt,
                 learn_rate=learn_rate,
                 batch_size=batch_size,
                 show=show)
    ft = sae.get_hidden_output(data_pretr)
    # print 'fts0', ft[rng.randint(0,ft.shape[0],(10,)),:]

    preds = []  # predictions
    for i_device in range(n_devices-1):
        preds += [np.zeros(y.shape)]

    # iterate for each cross-validation
    for i_cv in range(n_cross_val):
        mask_test = np.zeros((n_smp,), dtype=bool)
        mask_test[i_cv::n_cross_val] = True
        mask_train = ~mask_test

        x_labeled = ft_all[id_master][mask_train, :]
        y_train = y[mask_train, :]

        # supervised fine-tune the SAE
        sae.fine_tune_sup(x=x_labeled, y=y_train, loss_fun_name='mse',
                          max_iter=max_iter, opt_method=opt,
                          L1_reg=0.00, L2_reg=.0001,
                          learn_rate=learn_rate, batch_size=batch_size, show=show)
        # ft = sae.get_hidden_output(x_labeled)
        # print 'fts1', ft[rng.randint(0,ft.shape[0],(10,)),:]

        # domain ft for labeled smps
        n_train = x_labeled.shape[0]
        xl_domain_ft = np.zeros((n_train,n_devices))
        xl_domain_ft[:, id_master] = 1.

        # add ALL smp for reconstr
        x_all = ft_all[id_master]
        xa_domain_ft = np.zeros((n_smp,n_devices))
        xa_domain_ft[:, id_master] = 1.
        for i_device, id_device in enumerate(id_slave):
            x_all = np.vstack((x_all, ft_all[id_device]))
            d1 = np.zeros((n_smp, n_devices))
            d1[:, id_device] = 1.
            xa_domain_ft = np.vstack((xa_domain_ft, d1))

        # prepare transfer samples and their domain ft
        id_trans_in_train = trans_smp_list[:n_trans_smp, i_cv]
        tr_smp_src = np.tile(x_labeled[id_trans_in_train, :], (n_devices-1, 1))
        ts_domain_ft = np.zeros((n_trans_smp*(n_devices-1), n_devices))
        ts_domain_ft[:, id_master] = 1.

        tr_smp_tar = np.empty((0, n_ft))
        tt_domain_ft = np.zeros((n_trans_smp*(n_devices-1), n_devices))
        k = 0
        for id_device in id_slave:
            tmp = ft_all[id_device][mask_train, :]
            tr_smp_tar = np.vstack((tr_smp_tar, tmp[id_trans_in_train, :]))
            tt_domain_ft[k*n_trans_smp:(k+1)*n_trans_smp, id_device] = 1.
            k += 1

        tr_smp = (tr_smp_src, tr_smp_tar)
        tr_smp_domain_ft = (ts_domain_ft, tt_domain_ft)

        # train the DCAE
        dcae = Dcae(sae, n_domain_ft=n_devices,
                    use_correction_layer=use_correction_layer, rng=rng)
        dcae.set_model(n_labels=n_exp,
                       L1_reg=0.00, L2_reg=0.,
                       trans_reg=trans_reg, sup_reg=sup_reg,
                       sup_loss_fun_name='mse', pred_layer_act_fun_name='lin',
                       reconstr_reg=1.,
                       )
        dcae.train(x_labeled, y_train, xl_domain_ft,
                   x_all=x_all, xa_domain_ft=xa_domain_ft,
                   trans_smp_tuple=tr_smp,
                   tr_domain_ft_tuple=tr_smp_domain_ft,
                   max_iter=max_iter, opt_method=opt,
                   learn_rate=learn_rate, batch_size=batch_size, show=show)

        # get learned ft
        ft_train = dcae.get_hidden_output(x_labeled, xl_domain_ft)
        ft_test = []
        for i_device, id_device in enumerate(id_slave):
            d1 = np.zeros((n_smp-n_train, n_devices))
            d1[:, id_device] = 1.
            ft_test += [dcae.get_hidden_output(ft_all[id_device][mask_test, :], d1)]

        # prediction
        y_train = scaler_y.inverse_transform(y_train)
        run_reg(ft_train, ft_test, y_train, mask_test, preds)
        print "cross-validation %d/4 finished" % (i_cv+1)

    y_ori = scaler_y.inverse_transform(y)
    for i_device in range(n_devices-1):
        err[i_device, :] = np.sqrt(((preds[i_device]-y_ori)**2).mean(axis=0))

    # the 2 rows of err are error in 2 devices; column 1-4 are 4 regression labels
    # last column is average
    err = np.hstack((err, err.mean(axis=1).reshape((-1, 1))))
    print "error:", err


if __name__ == '__main__':
    n_trans_smp = 7
    corrupt = .1
    use_correction_layer = False
    n_hidden = [15]
    trans_reg = 16
    sup_reg = 2
    rng = np.random.RandomState(0)

    test()
