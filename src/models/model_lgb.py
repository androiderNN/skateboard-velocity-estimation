import os, sys
import numpy as np
import pandas as pd
import lightgbm as lgb

import model_base
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import config

class modeler_lgb(model_base.base):
    def __init__(self, split_by_subject=False, rand=0):
        super().__init__(split_by_subject=split_by_subject, rand=rand)

        self.params = {
            'objective': 'regression',
            'metric': 'rmse',
            'random_state': rand,
            'verbose': -1,
            'min_child_samples': 100,
            'bagging_fraction': 0.5,
            'bagging_freq': 1
        }

    def train_fn(self, tr_x, tr_y, va_x, va_y):
        '''
        trainとvalidのDataframeを投げるとlightgbmのモデルを返す'''        
        tr_lgb = lgb.Dataset(tr_x, tr_y)
        va_lgb = lgb.Dataset(va_x, va_y)

        model = lgb.train(
            params=self.params,
            train_set=tr_lgb,
            num_boost_round=1000,
            valid_sets=va_lgb,
            valid_names=['train', 'valid'],
            callbacks=[lgb.early_stopping(stopping_rounds=3, verbose=True)]
        )

        return model

    def predict(self, model, x:pd.DataFrame):
        return model.predict(x.drop(columns=config.drop_list, errors='ignore'))

    def main(self):
        super().main()
        
        # feature importance出力
        importance_df = pd.DataFrame( \
            {t: self.model[i].feature_importance(importance_type='gain') for i, t in enumerate(config.target_name)}, \
            columns=config.target_name)
        importance_df['mean'] = importance_df.mean(axis=1)
        importance_df.insert(0, 'index', self.train.drop(columns=config.drop_list, errors='ignore').columns)
        importance_df.sort_values('mean', ascending=False, inplace=True)
        print('\n', importance_df, '\n')

        if self.exornot:
            importance_df.to_csv(os.path.join(self.expath, 'importance.csv'))

if __name__=='__main__':
    modeler = modeler_lgb(split_by_subject=False, rand=1)
    modeler.main()    
