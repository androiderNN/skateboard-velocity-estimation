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

def iemg(data_myo, n_in_tr=6, n_for_ti=6, n_space=3):
    num_trial = data_myo.shape[0]

    # iemg計算
    ie = iemg_core(data_myo)
    
    # trialの記述
    # 1000個のデータから均等にn_in_tr点のデータを取得してtrial全体を描写する
    index = np.array([0] + [int(round((i+1)*1000/(n_in_tr-1)))-1 for i in range(n_in_tr-1)])
    ie_tr = ie[:, :, index].reshape(num_trial, -1)
    ie_tr_col = [c+'_iemgtr_'+str(index[i]) for c in config.feature_name for i in range(n_in_tr)]

    ie_tr = np.array([[l]*30 for l in ie_tr])
    ie_tr = ie_tr.reshape(-1, ie_tr.shape[2])

    ie_tr_df = pd.DataFrame(ie_tr, columns=ie_tr_col)
    ie_tr_df[['trial', 'timepoint']] = [[tr+1, ti] for tr in range(num_trial) for ti in range(30)]

    # 速度観測時点前後のiemgデータ抽出
    # 速度観測時点前後のiemgデータを1/n_space点ごとに前後各n_for_ti個取得する　データ数は2n+1個
    iemg_index = [round((i+1)*1000/30) for i in range(30)]  # 速度計測時刻の筋電位観測データインデックス
    iemg_index = [i if i-(n_for_ti*n_space)>0 else (n_for_ti*n_space) for i in iemg_index]  # インデックスが0を下回るときはminが0になるよう調整
    iemg_index = [i if i+(n_for_ti*n_space)<1000 else 1000-(n_for_ti*n_space)-1 for i in iemg_index]  # インデックスが1000を越えるときはmaxが1000になるよう調整
    iemg_index = [[i-(n_for_ti*n_space)+(n_space*j) for j in range(2*n_for_ti+1)] for i in iemg_index]  # indexの二次元配列を得る
    ie_ti = ie[:,:,iemg_index]
    ie_ti = ie_ti.transpose(0,2,1,3)
    ie_ti = ie_ti.reshape(num_trial*30, -1)
    ie_ti_col = [c+'_iemgti_'+str(i) for c in config.feature_name for i in iemg_index[0]]

    ie_ti_df = pd.DataFrame(ie_ti, columns=ie_ti_col)
    ie_ti_df[['trial', 'timepoint']] = [[tr+1, ti] for tr in range(num_trial) for ti in range(30)]

    ie_df = pd.merge(ie_tr_df, ie_ti_df, on=['trial', 'timepoint'])

    return ie_df