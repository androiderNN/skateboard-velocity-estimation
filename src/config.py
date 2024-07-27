import os

fdir = '../files'
train_path = os.path.join(fdir, 'train.mat')
test_path = os.path.join(fdir, 'test.mat')

target_name = ['vel_x', 'vel_y', 'vel_z']
feature_name = ['TA R', 'TA L', 'LG R', 'LG L', 'RF R', 'RF L', 'VL R', 'VL L', 'ST R', 'ST L', 'GMAX R', 'GMAX L', 'EMI R', 'EMI L', 'DEL R', 'DEL L']