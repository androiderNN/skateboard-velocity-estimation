import os, sys, pickle
import scipy.io as sio
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision.models.feature_extraction import create_feature_extractor

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import config
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))
import model_torch_base

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def extract_cnn():
    id = 'cnn_1014_14:13:42'
    trainers = pickle.load(open(os.path.join(config.saved_model_dir, id+'.pkl'), 'rb'))

    train = sio.loadmat(config.train_raw_path)
    test = sio.loadmat(config.test_raw_path)

    train_myo = model_torch_base.myo_processor(train, 5)
    test_myo = model_torch_base.myo_processor(test, 5)

    tr_features, te_features = list(), list()

    for i in range(3):
        model = trainers[i].modeler.model
        feature_extractor = create_feature_extractor(model, {'conv4': 'feature'})

        tr_fs = feature_extractor(torch.tensor(train_myo, dtype=torch.float32))['feature']
        te_fs = feature_extractor(torch.tensor(test_myo, dtype=torch.float32))['feature']

        tr_fs = tr_fs.reshape((tr_fs.shape[0], -1)).detach().numpy()
        te_fs = te_fs.reshape((te_fs.shape[0], -1)).detach().numpy()

        tr_features.append(tr_fs)
        te_features.append(te_fs)
    
    tr_features = np.array(tr_features).transpose(1,0,2).reshape((tr_features[0].shape[0], -1))
    te_features = np.array(te_features).transpose(1,0,2).reshape((te_features[0].shape[0], -1))

    return tr_features, te_features

def extract_from_nn():
    train_features = pickle.load(open(config.train_trial_path, 'rb')).loc[:, ['sid', 'trial']]
    test_features = pickle.load(open(config.test_trial_path, 'rb')).loc[:, ['sid', 'trial']]

    # nnから抽出
    tr_fs, te_fs = extract_cnn()

    # 結合
    num_fs = tr_fs.shape[1]
    cols = ['nn_fs_'+str(i) for i in range(num_fs)]
    
    tr_fs = pd.DataFrame(tr_fs, columns=cols)
    te_fs = pd.DataFrame(te_fs, columns=cols)

    train_features = train_features.join(tr_fs)
    test_features = test_features.join(te_fs)

    return train_features, test_features
