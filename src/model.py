import os, pickle, datetime, json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.metrics import mean_squared_error

import config, processing

rand = 0
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'random_state': rand,
    'verbose': -1,
}
# params.update({
#     'n_estimators': ,
#     'learning_rate': ,
#     'max_depth': ,
#     'num_leaves': ,
#     'min_child_samples': ,
#     'subsample': ,
#     'colsample_bytree': ,
#     'subsample_freq': 
# })

def lgb_train(tr_x, tr_y, va_x, va_y):
    '''
    trainとvalidのDataframeを投げるとlightgbmのモデルを返す'''
    tr_lgb = lgb.Dataset(tr_x, tr_y)
    va_lgb = lgb.Dataset(va_x, va_y)

    model = lgb.train(
        params=params,
        train_set=tr_lgb,
        num_boost_round=100,
        valid_sets=va_lgb,
        valid_names=['train', 'valid'],
        callbacks=[lgb.early_stopping(stopping_rounds=3, verbose=True)]
    )

    return model

def lgb_predict(model, x:pd.DataFrame):
    return model.predict(x.drop(columns=config.drop_list, errors='ignore'))

def print_score(tr_y, tr_pred, va_y, va_pred, va_es_y, va_es_pred):
    '''
    正解と予測値を渡すとスコアを出力する'''
    print('train :', mean_squared_error(tr_y, tr_pred)**0.5)
    print('estop :', mean_squared_error(va_es_y, va_es_pred)**0.5)
    print('valid :', mean_squared_error(va_y, va_pred)**0.5)

def make_submission(test):
    '''
    予測済のtest dataframeを投げると投稿ファイルを出力する
    jsonの形式に合わせるためintやlistへの変換を行っている'''
    dic = dict()

    for sub in range(4):
        tmp = test.loc[test['sid']==sub+1]
        sub = 'sub' + str(sub+1)
        dic[sub] = dict()

        for trial in np.unique(np.array(tmp['trial'])):
            dic[sub]['trial'+str(trial+1)] = [list(a) for a in np.array(tmp.loc[tmp['trial']==trial, [t+'_pred' for t in config.target_name]])]

    now = datetime.datetime.now()
    dirname = 'lgb_' + now.strftime('%m%d_%H:%M:%S')
    dirpath = os.path.join(config.exdir, dirname)
    os.mkdir(dirpath)   # 出力日時記載のフォルダ作成
    json.dump(dic, open(os.path.join(dirpath, 'submission.json'), 'w'))

    print('\nexport succeed')

def get_tr_va_index(train):
    '''
    trainのデータフレームを投げるとtrain, val, va_esのindexを作成'''
    sid = train['sid'].tolist()
    trial = train['trial'].tolist()
    train['trial_id'] = [str(sid[i])+'_'+str(trial[i]) for i in range(len(train))]    # sidとtrialでtrial_idを作成
    trial_id = np.unique(np.array(train['trial_id']))
    tr_id, va_id = train_test_split(trial_id, test_size=0.2, random_state=rand, shuffle=True)
    tr_id, va_es_id = train_test_split(tr_id, test_size=0.2, random_state=rand, shuffle=True)

    tr_index = [id in tr_id for id in train['trial_id']]
    va_index = [id in va_id for id in train['trial_id']]
    va_es_index = [id in va_es_id for id in train['trial_id']]
    train.drop(columns='trial_id', inplace=True)    # trainは参照なのでtrial_idを削除しておく

    return tr_index, va_index, va_es_index

def get_model_prediction(train, test, target):
    '''
    train, test, 予測対象を投げると予測値を追加したtestを返す'''
    x = train.drop(columns=config.drop_list, errors='ignore')
    y = train[target]

    # trainとvalidの分割
    tr_index, va_index, va_es_index = get_tr_va_index(train)
    tr_x, tr_y = x[tr_index], y[tr_index]
    va_x, va_y = x[va_index], y[va_index]
    va_es_x, va_es_y = x[va_es_index], y[va_es_index]

    # 学習と予測
    model = lgb_train(tr_x, tr_y, va_es_x, va_es_y)
    tr_pred = lgb_predict(model, tr_x)
    va_es_pred = lgb_predict(model, va_es_x)
    va_pred = lgb_predict(model, va_x)

    print_score(tr_y, tr_pred, va_y, va_pred, va_es_y, va_es_pred)
    return model

def rmse_3d(train:pd.DataFrame):
    '''
    3次元rmseの計算
    trainは予測値算出済'''
    sum_rmse = 0
    for sid in range(4):
        sid += 1
        tmp = train.loc[train['sid']==sid]
        se = np.array([(np.array(tmp[t]) - np.array(tmp[t+'_pred']))**2 for t in config.target_name])
        mse = np.sum(se) / se.shape[1]
        sum_rmse += mse**0.5

    mean_rmse = sum_rmse/4
    return mean_rmse

def main():
    train = pickle.load(open(config.train_pkl_path, 'rb'))
    test = pickle.load(open(config.test_pkl_path, 'rb'))

    # 予測後の変換
    # train = processing.convert_y(train)
    # test = processing.convert_y(test)

    # x, y, zごとのモデル作成と予測
    for target in config.target_name:
        print('\ntarget :', target)

        # 被験者ごとのモデル作成と予測
        # for sid in range(4):
        #     sid = sid + 1
        #     print('\n被験者id :', sid)
        #     train_tmp = train.loc[train['sid']==sid].copy()

        #     mod = get_model_prediction(train_tmp, test.loc[test['sid']==sid].copy(), target)
            
        #     test.loc[test['sid']==sid, target+'_pred'] = lgb_predict(mod, test.loc[test['sid']==sid].drop(columns=config.drop_list, errors='ignore'))
        #     train.loc[train['sid']==sid, target+'_pred'] = lgb_predict(mod, train_tmp)

        # 被験者で分割しない場合
        mod = get_model_prediction(train, test, target)
        test[target+'_pred'] = lgb_predict(mod, test.drop(columns=config.drop_list, errors='ignore'))
        train[target+'_pred'] = lgb_predict(mod, train.drop(columns=config.drop_list, errors='ignore'))

    # 予測後の変換
    # train = processing.convert_y(train)
    # test = processing.convert_y(test)

    rmse = rmse_3d(train)
    print('\nrmse :', rmse)

    i = input('出力しますか(y/n)')=='y'
    if i:
        make_submission(test)

if __name__=='__main__':
    main()
