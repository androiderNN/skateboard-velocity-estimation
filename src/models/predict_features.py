import pickle, sys, os, math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import config
from processing import process_core
sys.path.append(os.path.join(os.path.dirname(__file__)))
import model_lgb, model_base

train = pickle.load(open(config.train_path, 'rb'))
test = pickle.load(open(config.test_path, 'rb'))
tr_tr_df = pickle.load(open(config.train_trial_path, 'rb'))
te_tr_df = pickle.load(open(config.test_trial_path, 'rb'))

def rmse(y, pred):
    # pred = pred.argmax(axis=1)
    pred = [max(min(p, 29), 0) for p in pred]
    return model_base.rmse(y, pred)

def score_fn(y, pred):
    pred = [max(min(p, 29), 0) for p in pred]
    score = {
        'rmse': model_base.rmse(y, pred),
        'acc_5': round((abs(y-pred)<=5).sum()*100/len(y)),
        'acc_10': round((abs(y-pred)<=10).sum()*100/len(y))
    }
    return score

def x_min_predict(tr=None, te=None):
    tr_trial_df = tr_tr_df if tr is None else tr
    te_trial_df = te_tr_df if te is None else te

    train_array = np.array(train[['sid', 'trial', 'vel_x']])
    time_min_vel_x = list()

    # x座標が最大になる点
    for i in range(0, len(train), 30):
        coor = [0]  # x座標
        for t in range(30):
            coor.append(coor[-1] + train_array[i+t,2]/30)
        coor = np.array(coor[1:])   # 0点を除く

        time_min_vel_x.append([train_array[i,0], train_array[i,1], np.argmax(coor)])

    # 速度の絶対値が最小になる点
    # for i in range(0, len(train), 30):
    #     tmp = abs(train_array[i:i+30,2])
    #     time_min_vel_x.append([train_array[i,0], train_array[i,1], np.argmin(tmp)])

    time_min_df = pd.DataFrame(time_min_vel_x, columns=['sid', 'trial', 'x_min_time'])
    tr_trial_df = pd.merge(tr_trial_df, time_min_df, on=['sid', 'trial'])
    
    # colの定義
    col = [c for c in tr_trial_df.columns if c not in config.drop_list]
    col.remove('x_min_time')

    # index定義
    rand = 0

    # 0と30のデータ数を減らす
    # n = len(tr_trial_df)
    # m = 100
    # index = [i for i in range(n) if tr_trial_df.loc[i,'x_min_time']%29 != 0]   # 0,29以外のindex int
    # time_0_index = [i for i in range(n) if tr_trial_df.loc[i,'x_min_time']==0] # 0のindex
    # # index.extend(np.random.choice(time_0_index, size=m))
    # index.extend(time_0_index)
    # time_29_index = [i for i in range(n) if tr_trial_df.loc[i,'x_min_time']==29] # 29のindex
    # index.extend(np.random.choice(time_29_index, size=m))  # randomに30個選ぶ

    # idx = np.random.choice(index, size=round(len(index)*0.2))   # valid
    # index = [i for i in index if i not in idx]  # train
    # index = [[i in index for i in range(n)], [i in idx for i in range(n)]]

    # ランダム
    # n = len(tr_trial_df)
    # index = np.array([True for _ in range(n)])
    # index[np.random.choice(n, size=round(n*0.2), replace=False)] = False
    # index = [index, np.array([not i for i in index])]
    index = model_base.get_tr_va_index(tr_trial_df, es_size=0.2, rand=rand)

    # パラメータ定義
    # multiclass
    '''
    params = {
        'lgb_params': {
            'objective': 'multiclass',
            'num_class': 30,
            # 'metric': 'multi_logloss',
            'metric': 'multi_error',
            'random_state': rand,
            'verbose': -1,
            # 'max_depth': 10,
            # 'n_iter': 20,
            'bagging_fraction': 0.5,
            'bagging_rate': 1
        }
    }
    '''

    # regression
    params = {
        'lgb_params': {
            'objective': 'regression',  
            'metric': 'mse',
            'random_state': rand,
            'verbose': -1,
            'max_depth': 10,
            # 'n_iter': 20,
            # 'learning_rate': 0.1,
            'bagging_fraction': 0.5,
            'bagging_rate': 1
        }
    }

    trainer = model_base.holdout_training(model_lgb.modeler_lgb, params, score_fn=score_fn, rand=rand)
    # trainer = model_base.cv_training(model_lgb.modeler_lgb, params, score_fn=score_fn, rand=rand)

    preds = trainer.main(tr_trial_df, col, 'x_min_time', te_trial_df, index)

    # tr_trial_df['pred'] = preds[0].argmax(axis=1)
    # te_trial_df['pred'] = preds[1].argmax(axis=1)

    tr_trial_df['pred'] = [max(min(p, 29), 0) for p in preds[0]]
    te_trial_df['pred'] = [max(min(p, 29), 0) for p in preds[0]]

    return tr_trial_df, index
