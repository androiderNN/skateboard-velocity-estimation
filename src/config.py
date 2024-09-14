import os

fdir = '../data'
exdir = '../export'
train_path = os.path.join(fdir, 'train.mat')
test_path = os.path.join(fdir, 'test.mat')
train_pkl_path = os.path.join(fdir, 'train_df.pkl')
test_pkl_path = os.path.join(fdir, 'test_df.pkl')

target_name = ['vel_x', 'vel_y', 'vel_z']
# feature_name = ['TA R', 'TA L', 'LG R', 'LG L', 'RF R', 'RF L', 'VL R', 'VL L', 'ST R', 'ST L', 'GMAX R', 'GMAX L', 'EMI R', 'EMI L', 'DEL R', 'DEL L']
feature_name = ['TA R', 'TA L', 'LG R', 'LG L', 'RF R', 'RF L', 'VL R', 'VL L', 'ST R', 'ST L', 'GM R', 'GM L', 'EM R', 'EM L', 'DE R', 'DE L']
# drop_list = ['sid', 'trial', 'timepoint', 'vel_x', 'vel_y', 'vel_z', 'vel_x_pred', 'vel_y_pred', 'vel_z_pred']
drop_list = ['isregular', 'trial', 'vel_x', 'vel_y', 'vel_z', 'vel_x_pred', 'vel_y_pred', 'vel_z_pred']

drop_list = drop_list + feature_name