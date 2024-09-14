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

    ie_ti = iemg.iemg(data_myo)
    data_df = pd.merge(data_df, ie_ti, on=['trial', 'timepoint'])
    
    # fft計算
    # fft_df = fft.fft_onVelosityTime(data_myo)
    # data_df = pd.merge(data_df, fft_df, on=['trial', 'timepoint'])

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
