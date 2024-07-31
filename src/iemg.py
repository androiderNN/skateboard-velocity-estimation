import numpy as np
import pandas as pd
from scipy import signal

fs = 2000
lowcut = 4.8
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
    return y

def iemg(data_myo):
    data_myo = abs(data_myo)
    data_myo = np.apply_along_axis(apply_filter, 2, data_myo)
    return data_myo
