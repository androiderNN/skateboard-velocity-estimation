import os, pickle
import numpy as np
import pandas as pd
from scipy import io as sio
from scipy import signal

import config

train_raw = sio.loadmat(config.train_path)
test_raw = sio.loadmat(config.test_path)
vel_sample_rate = 30    # 30fps

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

def process(data_skater:np.array, isregular:bool, sid:int):
    '''
    被験者一人当たりのデータを入力するとモデルに入力可能な形式に変換する
    筋電位データと体勢データを展開'''
    data_myo = data_skater[0,0][0]
    data_extracted = np.array([])

    for arr in data_myo:    # trialごとに分割
        data_extracted = np.append(data_extracted, np.array([signal.resample(dm, vel_sample_rate) for dm in arr]).T)
    
    data_extracted = data_extracted.reshape(16, -1).T   # np.appendでflatになったarrayを2次元に復元
    data_df = pd.DataFrame(data_extracted, columns=config.feature_name)
    data_df['isregular'] = int(isregular)

    data_df['sid'] = sid
    data_df['trial'] = [t for _ in range(vel_sample_rate) for t in range(len(data_myo))]
    data_df['timepoint'] = [i for i in range(vel_sample_rate) for _ in range(len(data_myo))]
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
            tmp_df[config.target_name] = raw_data[sid][0,0][1].transpose(1,2,0).reshape(3, -1).T
        
        df = pd.concat([df, tmp_df])
    
    df.reset_index(inplace=True, drop=True)
    return df

if __name__ == '__main__':
    train_df = make_data(train_raw, True)
    test_df = make_data(test_raw, False)

    pickle.dump(train_df, open(config.train_pkl_path, 'wb'))
    pickle.dump(test_df, open(config.test_pkl_path, 'wb'))

    print('successfully finished.')
