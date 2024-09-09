import os, pickle
import numpy as np
import pandas as pd
from scipy import io as sio
from scipy import signal

import config
from features import iemg, fft

train_raw = sio.loadmat(config.train_path)
test_raw = sio.loadmat(config.test_path)

def convert_y(df, goofy_only:bool):
    '''
    データを左右対称に入れ替える関数'''
    if goofy_only:
        df_tmp = df.loc[df['isregular']==0]
    else:
        df_tmp = df

    # 列名の変更により筋電位データの左右入れ替え
    col = [c[:-1]+c[-1].translate(str.maketrans({'L':'R', 'R':'L'})) for c in df_tmp.columns]
    df_tmp.columns = col
    
    # 速度の実測値・予測値の正負を入れ替え
    if 'vel_y' in col:
        df_tmp.loc[:, 'vel_y'] = df_tmp['vel_y']*-1
    if 'vel_y_pred' in col:
        df_tmp.loc[:, 'vel_y_pred'] = df_tmp['vel_y_pred']*-1
    
    if goofy_only:
        df.loc[df['isregular']==0] = df_tmp
    else:
        df = df_tmp
    
    return df

def process(data_sub:np.array, isregular:bool, sid:int):
    '''
    被験者一人当たりのデータを入力するとモデルに入力可能な形式に変換する
    筋電位データと体勢データを展開'''
    data_myo = data_sub[0,0][0]  # 筋電位データ
    num_trial = data_myo.shape[0]

    # 基本列作成
    data_df = pd.DataFrame([[t+1, i] for t in range(num_trial) for i in range(30)], columns=['trial', 'timepoint'])
    data_df['isregular'] = int(isregular)
    data_df.insert(0, 'sid', sid)

    '''
    # iemg計算
    ie = iemg.iemg(data_myo)
    
    # trialの記述
    # 1000個のデータから均等にn点のデータを取得してtrial全体を描写する
    n = 6
    index = np.array([0] + [int(round((i+1)*1000/(n-1)))-1 for i in range(n-1)])
    ie_tr = ie[:, :, index].reshape(num_trial, -1)
    ie_tr_col = [c+'_itrial_'+str(index[i]) for c in config.feature_name for i in range(n)]

    ie_tr = pd.DataFrame(ie_tr, columns=ie_tr_col)
    ie_tr['trial'] = [i for i in range(num_trial)]
    data_df = pd.merge(data_df, ie_tr, on='trial')

    # 速度観測時点前後のiemgデータ挿入
    # 速度観測時点前後のiemgデータを1/m点ごとに前後各n個取得する　データ数は2n+1個
    n = 10
    m = 3
    iemg_index = [round((i+1)*1000/30) for i in range(30)]  # 速度計測時刻の筋電位観測データインデックス
    iemg_index = [i if i-(n*m)>0 else (n*m) for i in iemg_index]  # インデックスが0を下回るときはminが0になるよう調整
    iemg_index = [i if i+(n*m)<1000 else 1000-(n*m)-1 for i in iemg_index]  # インデックスが1000を越えるときはmaxが1000になるよう調整
    iemg_index = [[i-(n*m)+(m*j) for j in range(2*n+1)] for i in iemg_index]  # indexの二次元配列を得る
    ie_ti = ie[:,:,iemg_index]
    ie_ti = ie_ti.transpose(0,2,1,3)
    ie_ti = ie_ti.reshape(num_trial*30, -1)
    ie_ti_col = [c+'_itime_'+str(i) for c in config.feature_name for i in iemg_index[0]]

    ie_ti = pd.DataFrame(ie_ti, columns=ie_ti_col)
    ie_ti[['trial', 'timepoint']] = [[tr, ti] for tr in range(num_trial) for ti in range(30)]
    data_df = pd.merge(data_df, ie_ti, on=['trial', 'timepoint'])
    '''
    
    # fft計算
    fft_df = fft.fft_onVelosityTime(data_myo)
    data_df = pd.merge(data_df, fft_df, on=['trial', 'timepoint'])

    return data_df

def make_data(raw_data:np.array, istrain:bool):
    '''
    rawデータを投げるとDataFrameに変換する
    trainデータの場合は速度列も追加して返す'''
    df = pd.DataFrame()

    for i in range(4):  # 被験者ごとにprocessに投げる
        sid = '000' + str(i+1)
        tmp_df = process(raw_data[sid], raw_data[sid][0,0][-2][0]=='regular', i+1)

        # trainならば速度列をDataframeに追加する
        if istrain:
            tmp_df[config.target_name] = raw_data[sid][0,0][1].reshape(-1,3,30).transpose(0,2,1).reshape(-1,3)
        
        df = pd.concat([df, tmp_df])
    
    df.reset_index(inplace=True, drop=True)
    return df

if __name__ == '__main__':
    train_df = make_data(train_raw, True)
    test_df = make_data(test_raw, False)

    pickle.dump(train_df, open(config.train_pkl_path, 'wb'))
    pickle.dump(test_df, open(config.test_pkl_path, 'wb'))

    print('successfully finished.')
