import os, sys
import numpy as np
import pandas as pd
from scipy import signal

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import config

fs = 2000
lowcut = 5
order = 2

def get_filter(lowcut, fs, order):
    '''
    ローパスフィルタを設定する関数'''
    nyq = 0.5 * fs
    low = lowcut / nyq
    b, a = signal.butter(order, low, btype='low')
    return b, a

def apply_filter(x):
    '''
    フィルタを適用する関数'''
    b, a = get_filter(lowcut, fs, order)
    y = signal.filtfilt(b, a, x)
    y = y/y.max()   # 最大値で除して正規化
    return y

def iemg_core(data_myo):
    '''
    一人分の筋電位データを入力するとフィルタ適用後の筋電位データを出力する'''
    data_myo = abs(data_myo)
    data_myo = np.apply_along_axis(apply_filter, 2, data_myo)
    return data_myo

def pickfromtrial(ie, n):
    '''
    1000個のデータから均等にn点のデータを取得してtrial全体を描写する'''
    num_trial = ie.shape[0]

    index = np.array([0] + [int(round((i+1)*1000/(n-1)))-1 for i in range(n-1)])
    ie_tr = ie[:, :, index].reshape(num_trial, -1)
    ie_tr_col = [c+'_iemgtr_'+str(index[i]) for c in config.feature_name for i in range(n)]

    ie_tr = np.array([[l]*30 for l in ie_tr])
    ie_tr = ie_tr.reshape(-1, ie_tr.shape[2])

    df = pd.DataFrame(ie_tr, columns=ie_tr_col)
    df[['trial', 'timepoint']] = [[tr+1, ti] for tr in range(num_trial) for ti in range(30)]
    return df

def pickfortimepoint(ie, n, m):
    '''
    速度観測時点前後のiemgデータをm点ごとに前後各n個取得する データ数は2n+1個'''
    num_trial = ie.shape[0]

    iemg_index = [round((i+1)*1000/30) for i in range(30)]  # 速度計測時刻の筋電位観測データインデックス
    iemg_index = [i if i-(n*m)>0 else (n*m) for i in iemg_index]  # インデックスが0を下回るときはminが0になるよう調整
    iemg_index = [i if i+(n*m)<1000 else 1000-(n*m)-1 for i in iemg_index]  # インデックスが1000を越えるときはmaxが1000になるよう調整
    iemg_index = [[i-(n*m)+(m*j) for j in range(2*n+1)] for i in iemg_index]  # indexの二次元配列を得る
    ie_ti = ie[:,:,iemg_index]
    ie_ti = ie_ti.transpose(0,2,1,3)
    ie_ti = ie_ti.reshape(num_trial*30, -1)
    ie_ti_col = [c+'_iemgti_'+str(i) for c in config.feature_name for i in iemg_index[0]]

    df = pd.DataFrame(ie_ti, columns=ie_ti_col)
    df[['trial', 'timepoint']] = [[tr+1, ti] for tr in range(num_trial) for ti in range(30)]
    return df

def iemg(data_myo):
    num_trial = data_myo.shape[0]
    df = pd.DataFrame([[t+1, i] for t in range(num_trial) for i in range(30)], columns=['trial', 'timepoint'])

    n_pick_tr = 6
    n_pick_ti = 6
    n_space_ti = 3

    # iemg計算
    ie = iemg_core(data_myo)
    
    # trialの記述
    tmp_df = pickfromtrial(ie, n_pick_tr)
    df = pd.merge(df, tmp_df, on=['trial', 'timepoint'])

    # 速度観測時点前後のiemgデータ抽出
    tmp_df = pickfortimepoint(ie, n_pick_ti, n_space_ti)
    df = pd.merge(df, tmp_df, on=['trial', 'timepoint'])

    return df