import os

fdir = '../files'
exdir = '../export'
train_path = os.path.join(fdir, 'train.mat')
test_path = os.path.join(fdir, 'test.mat')
train_pkl_path = os.path.join(fdir, 'train_df.pkl')
test_pkl_path = os.path.join(fdir, 'test_df.pkl')

target_name = ['vel_x', 'vel_y', 'vel_z']
feature_name = ['TA R', 'TA L', 'LG R', 'LG L', 'RF R', 'RF L', 'VL R', 'VL L', 'ST R', 'ST L', 'GMAX R', 'GMAX L', 'EMI R', 'EMI L', 'DEL R', 'DEL L']
drop_list = ['sid', 'trial', 'timepoint', 'vel_x', 'vel_y', 'vel_z']
