cornData.mat:

http://www.eigenvector.com/data/Corn/

ftAll		the 80 corn samples collected with 3 devices
Y			the 4 target values of the 80 samples
trList		the indices of 50 pre-selected transfer samples for the 4-fold cross validation
We are using device 1 as the master device to transfer to devices 2 and 3


UciData.mat

http://archive.ics.uci.edu/ml/datasets/Gas+Sensor+Array+Drift+Dataset+at+Different+Concentrations

data		original samples in 10 batches
label		class labels in 10 batches (6 kinds of gases)
concentr	gas concentration in 10 batches in ppm
gasNames	the names of the gases
smpStat 	number of samples in each batch and each class
Tsrc			source transfer samples 
Ttar			target transfer samples
Because we wish to train on batch 1 (source domain) and test on batch 2 to 10 (target domains), we need to find a group of transfer samples in each batch b (b>1) which has a corresponding group in batch 1. Therefore, Ttar{i} is the group in batch i-1, whose corresponding group is in Tsrc{i}. That is to say, all groups in Tsrc are from batch 1, but correspond to different target batch. The method of selection is introduced in: K. Yan and D. Zhang, ¡°Calibration transfer and drift compensation of e-noses via coupled task learning,¡± Sens. Actuators B: Chem., vol. 225, pp. 288¨C297, 2016.