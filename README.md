# yan-autoencoder
Yet ANother autoencoder toolbox based on Theano

We implemented the autoencoder (AE) algorithm in Python based on the Theano library. The materials in http://deeplearning.net/ helped a lot.

yan_ae.py: 		the shallow AE. Various strategies can be enabled or disabled, including denoising, sparsity, L1/L2 regularization, and weight tying strategies. One can also config the loss function, activation function, and training algorithm (e.g. minibatch, CG).

yan_sae.py:		the stacked AE (SAE), built upon the shallow ones. Methods include layer-wise pretraining, unsupervised and supervised fine-tuning.

yan_mlp.py:		multi-layer perceptron (fully connected network), mainly implemented for fine-tuning of SAE.

yan_msda.py:	marginalized denoising autoencoder.

dcae.py:		drift correction autoencoder (DCAE), designed for correction of instrumental variation and time-varying drift in sensor systems. Please see the ref for details.

test_ae.py:		test SAE and MSDA on benchmark datasets.

test_corn.py:	test DCAE on the corn dataset.

test_uci.py:	test DCAE on the gas sensor array drift dataset. Description of these two datasets is in datasets_readme.txt.

.idea:			project files of PyCharm.

ref: Ke Yan, and David Zhang, "Correcting Instrumental Variation and Time-varying Drift: A Transfer Learning Approach with Autoencoders, " accepted by Instrumentation and Measurement, IEEE Transactions on

Copyright 2016 YAN Ke, Tsinghua Univ. http://yanke23.com , xjed09@gmail.com