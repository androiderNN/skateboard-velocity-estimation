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

def compress(df, cols, n_components, colname=''):
    model = PCA(n_components)
    compressed_data = model.fit_transform(df[cols])
    df.drop(columns=cols, inplace=True)
    # df[[colname+'_comp_'+str(i) for i in range(n_components)]] = compressed_data
    df = pd.concat([df, pd.DataFrame(compressed_data, columns=[colname+'_comp_'+str(i) for i in range(n_components)])], axis=1)
    return df
