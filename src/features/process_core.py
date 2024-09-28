import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def onehot(x):
    '''
    one-hot encodigする関数
    一列分のデータを入力するとデータ数*クラス数のndarrayを返す'''
    x = np.array(x)
    cl = np.unique(x)
    array = np.zeros(shape=(x.shape[0], cl.shape[0]))

    for i, c in enumerate(cl):
        array[x==c, i] = 1

    cl = [str(c) for c in cl]
    return array, cl

def compress(train, test, cols, n_components, colname='fft'):
    train = train.copy()
    test = test.copy()

    if len(cols)!=16*31:
        print('対象のcolumnが間違っている可能性あり')

    model = PCA(n_components)
    model.fit(train[cols])

    for df in [train, test]:
        compressed = model.transform(df[cols])
        df.drop(columns=cols, inplace=True)
        df = pd.concat([df, pd.DataFrame(compressed, columns=[colname+'_comp_'+str(i) for i in range(n_components)])], axis=1)

    # print(pd.DataFrame(model.explained_variance_ratio_).cumsum())

    return train, test

def normalize(train, test, cols):
    mean = train[cols].mean()
    std = train[cols].std()

    train[cols] = (train[cols] - mean) / std
    test[cols] = (test[cols] - mean) / std

    return train, test
