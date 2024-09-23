import os, sys
import numpy as np
import pandas as pd
import lightgbm as lgb

import model_base
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import config

class modeler_lgb(model_base.modeler_base):
    def __init__(self, params, rand, verbose=True):
        self.model = None
        self.params = params
        self.verbose = verbose

    def train(self, tr_x, tr_y, es_x, es_y):
        tr_lgb = lgb.Dataset(tr_x, tr_y)
        es_lgb = lgb.Dataset(es_x, es_y)

        self.model = lgb.train(
            params=self.params['lgb_params'],
            train_set=tr_lgb,
            num_boost_round=1000,
            valid_sets=es_lgb,
            valid_names=['train', 'estop'],
            callbacks=[lgb.early_stopping(stopping_rounds=3, verbose=self.verbose)]
        )
    
    def predict(self, x):
        return self.model.predict(x)

if __name__=='__main__':
    rand = 0
    params = {
        'rand': rand,
        'use_cv': False,
        'verbose': False,
        'lgb_params': {
            'objective': 'regression',
            'metric': 'rmse',
            'random_state': rand,
            'verbose': -1,
            'min_child_samples': 100,
            'bagging_fraction': 0.5,
            'bagging_freq': 1
        }
    }

    ins = model_base.vel_prediction(modeler_lgb, params=params, rand=rand, use_cv=False)
    ins.main()
