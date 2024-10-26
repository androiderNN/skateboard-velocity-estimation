import os, sys, pickle
import numpy as np
import pandas as pd
from scipy import signal
from sklearn.linear_model import LinearRegression
import scipy.io as sio

sys.path.append(os.path.dirname(__file__))
import fft
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

def apply_filter(x, lowcut=lowcut, fs=fs, order=order):
    '''
    フィルタを適用する関数'''
    b, a = get_filter(lowcut, fs, order)
    y = signal.filtfilt(b, a, x)
    return y

def iemg_core(data_myo, lowcut=lowcut):
    '''
    一人分の筋電位データを入力するとフィルタ適用後の筋電位データを出力する'''
    data_myo = abs(data_myo)
    # data_myo = np.apply_along_axis(apply_filter, 2, data_myo)
    data_myo = np.array([[apply_filter(data_myo[i,j,:], lowcut) for j in range(data_myo.shape[1])] for i in range(data_myo.shape[0])])
    data_myo = data_myo / data_myo.max(axis=2)[:,:,np.newaxis]  # 最大値で除して正規化
    return data_myo

def pickfromtrial(ie, n):
    '''
    1000個のデータから均等にn点のデータを取得してtrial全体を描写する'''
    num_trial = ie.shape[0]

    index = np.array([0] + [int(round((i+1)*1000/(n-1)))-1 for i in range(n-1)])
    ie_tr = ie[:, :, index].reshape(num_trial, -1)
    ie_tr_col = ['iemg_'+c+str(index[i]+1) for c in config.feature_name for i in range(n)]

    # ie_tr = np.array([l for l in ie_tr])
    # ie_tr = ie_tr.reshape(-1, ie_tr.shape[2])

    df = pd.DataFrame(ie_tr, columns=ie_tr_col)
    df['trial'] = [tr+1 for tr in range(num_trial)]
    return df

def describeiemg(ie):
    '''
    iemgから複数の特徴量を抽出する'''
    # 傾き、決定係数、ピーク数、最大位置、平均
    # features = ['iemg_coef', 'R^2', 'num_peaks', 'peak_posit', 'mean']
    features = ['iemg_coef', 'iemg_R^2', 'iemg_peakposit', 'iemg_mean']
    tmp = np.zeros(shape=(ie.shape[0], ie.shape[1], len(features)), dtype=np.float32)

    regressor = LinearRegression()
    x = np.linspace(0, 999, 1000, dtype=np.int32)[:, np.newaxis]

    for trial in range(ie.shape[0]):
        for col in range(ie.shape[1]):
            # 傾き、決定係数
            y = ie[trial, col, :]
            regressor.fit(x, y)
            tmp[trial, col, :2] = [regressor.coef_[0], regressor.score(x, y)]

            tmp[trial, col, 2] = ie[trial, col, :].mean()       # 平均
            tmp[trial, col, 3] = np.argmax(ie[trial, col, :])   # 最大値の位置
            # tmp[trial, col, 4] = len(signal.find_peaks(y, height=0)[0]) # ピーク数
    
    tmp = tmp.reshape((tmp.shape[0], -1))   # colと特徴量を同一次元に
    # tmp = np.array([[l]*30 for l in tmp]) # 30回分増幅
    # tmp = tmp.reshape((-1, tmp.shape[2]))
    df_col = [c+'_'+f for f in config.feature_name for c in features]
    df = pd.DataFrame(tmp, columns=df_col)
    df['trial'] = [t+1 for t in range(ie.shape[0])]
    return df

def fft_iemg(ie):
    '''
    フィルタ後のiemgをフーリエ変換する'''
    ie_fft = fft.fft_core(ie)[:,:,:2]  # ローパスフィルタをかけているのでフーリエ変換後2,4Hzのみ抜きだす
    ie_fft = ie_fft.reshape(ie_fft.shape[0], -1)    # (trial, features*2)

    ie_fft_df = pd.DataFrame(ie_fft, columns=['iemg_'+f+'_'+str(i) for f in config.feature_name for i in range(2,5,2)])
    ie_fft_df['trial'] = [i+1 for i in range(ie_fft.shape[0])]
    return ie_fft_df

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
    ie_ti_col = ['iemgti_'+c+'_'+str(i) for c in config.feature_name for i in iemg_index[0]]

    df = pd.DataFrame(ie_ti, columns=ie_ti_col)
    df[['trial', 'timepoint']] = [[tr+1, ti] for tr in range(num_trial) for ti in range(30)]
    return df

def iemg(data_myo, ie):
    num_trial = data_myo.shape[0]
    df = pd.DataFrame([t+1 for t in range(num_trial)], columns=['trial'])

    n_pick_tr = 6
    n_pick_ti = 6
    n_space_ti = 3

    # iemg計算
    # ie = iemg_core(data_myo)
    
    # trialの記述
    tmp_df = pickfromtrial(ie, n_pick_tr)
    df = pd.merge(df, tmp_df, on='trial')

    tmp_df = describeiemg(ie)
    df = pd.merge(df, tmp_df, on='trial')

    # tmp_df = fft_iemg(ie)
    # df = pd.merge(df, tmp_df, on='trial')

    # 速度観測時点前後のiemgデータ抽出
    # tmp_df = pickfortimepoint(ie, n_pick_ti, n_space_ti)
    # df = pd.merge(df, tmp_df, on=['trial', 'timepoint'])

    return df

def dump_iemg():
    train = sio.loadmat(config.train_raw_path)
    test = sio.loadmat(config.test_raw_path)

    l = list()
    for i in range(4):
        l.append(iemg_core(train['000'+str(i+1)][0,0][0]))
        
    pickle.dump(l, open(config.iemg_train_path, 'wb'))

    l = list()
    for i in range(4):
        l.append(iemg_core(test['000'+str(i+1)][0,0][0]))
        
    pickle.dump(l, open(config.iemg_test_path, 'wb'))
