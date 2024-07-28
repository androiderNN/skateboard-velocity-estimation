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

def process(data_sub:np.array, isregular:bool, sid:int):
    '''
    被験者一人当たりのデータを入力するとモデルに入力可能な形式に変換する
    筋電位データと体勢データを展開'''
    data_myo = data_sub[0,0][0]  # 筋電位データ
    data_extracted = np.array([])

    # 速度計測時点での筋電位データ
    for arr in data_myo:    # trialごとに分割
        data_extracted = np.append(data_extracted, np.array([signal.resample(dm, vel_sample_rate) for dm in arr]).T)
    
    data_extracted = data_extracted.reshape(16, -1).T   # np.appendでflatになったarrayを2次元に復元
    data_df = pd.DataFrame(data_extracted, columns=config.feature_name)

    # 各種タグ付け
    data_df['isregular'] = int(isregular)
    data_df['sid'] = sid
    data_df['trial'] = [t for _ in range(vel_sample_rate) for t in range(len(data_myo))]
    data_df['timepoint'] = [i for i in range(vel_sample_rate) for _ in range(len(data_myo))]

    # trialの記述
    data_myo_abs = abs(data_myo)
    data_myo_over02 = np.sum(data_myo_abs>0.2, axis=2)  # 筋電位の絶対値が0.2を越える計測値の個数
    data_myo_over01 = np.sum(data_myo_abs>0.1, axis=2)  # 0.1
    data_myo_over005 = np.sum(data_myo_abs>0.05, axis=2)    # 0.05

    data_myo_trial = np.concatenate([data_myo_over02, data_myo_over01, data_myo_over005], axis=1)
    col = [c+i for i in ['_ov02', '_ov01', 'ov005'] for c in config.feature_name]
    data_myo_trial = pd.DataFrame(data_myo_trial, columns=col)  # 新規DataFrame作成
    data_myo_trial['trial'] = [i for i in range(len(data_myo_trial))]

    # 筋電位データのdfとtrialの概略dfをtrialをキーに結合
    data_df = pd.merge(data_df, data_myo_trial, on='trial')

    # 左右の差分を特徴量に追加
    fs = np.unique(np.array([f[:-1] for f in config.feature_name]))
    arr = np.array(data_df[[f+'R' for f in fs]]) - np.array(data_df[[f+'L' for f in fs]])
    diff_df = pd.DataFrame(arr, columns=[f+'diff' for f in fs])
    data_df = pd.merge(data_df, diff_df, left_index=True, right_index=True)

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
