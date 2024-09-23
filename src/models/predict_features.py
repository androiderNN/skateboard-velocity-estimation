import pickle, sys, os, random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import config
sys.path.append(os.path.join(os.path.dirname(__file__)))
# import model_lgb
from . import model_lgb, model_base

train = pickle.load(open(config.train_path, 'rb'))
test = pickle.load(open(config.test_path, 'rb'))
tr_tr_df = pickle.load(open(config.train_trial_path, 'rb'))
te_tr_df = pickle.load(open(config.test_trial_path, 'rb'))

def rmse(y, pred):
    # pred = pred.argmax(axis=1)
    pred[pred<0] = 0
    pred[pred>=30] = 29
    return model_base.rmse(y, pred)

def x_min_predict():
    train_array = np.array(train[['sid', 'trial', 'vel_x']])
    time_min_vel_x = list()

    for i in range(0, len(train), 30):
        # x座標が最大になる点を探す 速度の方がよいか？
        coor = [0]  # x座標
        for t in range(30):
            coor.append(coor[-1] + train_array[i+t,2]/30)
        coor = np.array(coor[1:])   # 0点を除く

        time_min_vel_x.append([train_array[i,0], train_array[i,1], np.argmax(coor)])

    time_min_df = pd.DataFrame(time_min_vel_x, columns=['sid', 'trial', 'x_min_time'])
    tr_trial_df = pd.merge(tr_tr_df, time_min_df, on=['sid', 'trial'])

    # colとindexの定義
    te_trial_df = te_tr_df
    col = [c for c in tr_trial_df.columns if c not in config.drop_list]
    col.remove('x_min_time')

    n = len(tr_trial_df)
    index = np.array([True for _ in range(n)])
    index[np.random.choice(n, size=round(n*0.2), replace=False)] = False
    index = [index, np.array([not i for i in index])]

    rand = 0

    # params = {
    #     'lgb_params': {
    #         'objective': 'multiclass',
    #         'num_class': 30,
    #         # 'metric': 'multi_logloss',
    #         'metric': 'multi_error',
    #         'random_state': rand,
    #         'verbose': -1,
    #         # 'max_depth': 10,
    #         # 'n_iter': 20,
    #         'bagging_fraction': 0.5,
    #         'bagging_rate': 1
    #     }
    # }

    params = {
        'lgb_params': {
            'objective': 'regression',
            'metric': 'mse',
            'random_state': rand,
            'verbose': -1,
            # 'max_depth': 10,
            # 'n_iter': 20,
            # 'learning_rate': 0.1,
            'bagging_fraction': 0.6,
            'bagging_rate': 1
        }
    }

    trainer = model_base.holdout_training(model_lgb.modeler_lgb, params, score_fn=rmse, rand=rand)
    # trainer = model_base.cv_training(model_lgb.modeler_lgb, params, score_fn=rmse, rand=rand)
    preds = trainer.main(tr_trial_df, col, index, 'x_min_time', te_trial_df)

    # tr_trial_df['pred'] = preds[0].argmax(axis=1)
    # te_trial_df['pred'] = preds[1].argmax(axis=1)

    tr_trial_df['pred'] = preds[0]
    te_trial_df['pred'] = preds[1]

    return tr_trial_df
