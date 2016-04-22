# coding=utf-8
"""
Test the autoencoders in this toolbox (except DCAE) on benchmark datasets.

Please cite: Ke Yan, and David Zhang, "Correcting Instrumental Variation
and Time-varying Drift: A Transfer Learning Approach with Autoencoders,"
accepted by Instrumentation and Measurement, IEEE Transactions on

Copyright 2016 YAN Ke, Tsinghua Univ. http://yanke23.com , xjed09@gmail.com

"""
from sklearn import datasets
from sklearn import svm
from sklearn import preprocessing
from sklearn import linear_model
import matplotlib.pyplot as plt

from yan_sae import YanSae
from yan_msda import YanMsda


# import data
dataset = datasets.load_iris()
dataset = datasets.load_digits()

x = dataset.data
y = dataset.target
x_train = x[::2]
y_train = y[::2]
x_test = x[1::2]
y_test = y[1::2]

# preprocess
scaler = preprocessing.MinMaxScaler().fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# raw feature
clsf_model = linear_model.LogisticRegression(solver='liblinear', multi_class='ovr')
# clsf_model = svm.SVC(C=1., kernel='linear')
clsf_model.fit(x_train, y_train)
print 'Raw feature:', clsf_model.score(x_test, y_test)

# pretrain SAE
max_iter = 100
sae = YanSae(n_visible=x.shape[1],
             n_hidden=[60],
             act_fun_names=['lin'],
             corruption_levels=[.0],
             sparsity_reg=.00,
             sparsity_target=.1,
             L1_reg=.0,
             L2_reg=.0,
             use_tied_weight=True,
             use_biases=True)
sae.pretrain(x=x_train,
             max_iter=max_iter,
             show=[True]*3)
ft_train = sae.get_hidden_output(x_train)
ft_test = sae.get_hidden_output(x_test)
clsf_model.fit(ft_train, y_train)
print 'Accuracy of sae feature after pretrain:', clsf_model.score(ft_test, y_test)

# fine-tune
sae.fine_tune_sup(x=x_train,
                  y=y_train,
                  loss_fun_name='ce',
                  L1_reg=.0,
                  L2_reg=0.0,
                  max_iter=max_iter,
                  show=[True] * 3)
ft_train = sae.get_hidden_output(x_train)
ft_test = sae.get_hidden_output(x_test)
clsf_model.fit(ft_train, y_train)
print 'Accuracy of sae feature after finetuning:', clsf_model.score(ft_test, y_test)

# test marginalized SDA
sae = YanMsda()
sae.train(x_train)
ft_train = sae.get_hidden_output(x_train)
ft_test = sae.get_hidden_output(x_test)
clsf_model.fit(ft_train, y_train)
print 'Accuracy of MSDA feature:', clsf_model.score(ft_test, y_test)

plt.show()  # show the plot of the cost
