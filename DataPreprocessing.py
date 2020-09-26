#!/usr/bin/env python
# coding: utf-8

#__Author__ = "Chen Zhang"
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler, StandardScaler

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import lightgbm as lgb

#####################################################################################################

class DataInit(object):
    def __init__(self, train_raw, target_raw, cat_list, dropping_list=[], sc=StandardScaler(), encoder=OrdinalEncoder()):
        #self.catalist = cat_list
        self.enc = encoder
        self.sc = sc
        self.dropping_list = dropping_list
        self.train = self._cat_encoder(train_raw, cat_list)
        self.target = self._cat_encoder(target_raw, cat_list)
        self._init_feature_dropping(self.dropping_list)
        self._x_y_split()
        self._num_nan_fill()
        self._num_scaling()
        
    def _cat_encoder(self, raw_data, cat_list):
        #fill space for NaN of categorical features.
        raw_data.iloc[:,cat_list] = raw_data.iloc[:,cat_list].replace(np.nan, ' ', regex=True).astype(str)
        raw_data.iloc[:,cat_list] = self.enc.fit_transform(raw_data.iloc[:,cat_list]).astype(str)
        data = raw_data
        return data
    
    def _init_feature_dropping(self, drop_list):
        if type(drop_list[0]) == str:
            d_col_names = drop_list
        else:
            d_col_names = self.train.columns[drop_list]
        self.train = self.train.drop(d_col_names, axis=1)
        self.target = self.target.drop(d_col_names, axis=1)
    
    def _num_scaling(self):
        self.num_list = []
        for i in range(len(list(self.train_x))):
            if self.train_x.iloc[:,i].dtypes == float:
                self.num_list.append(self.train_x.columns[i])
            else:
                pass
        self.cat_list = []
        for i in range(len(list(self.train_x))):
            if self.train_x.iloc[:,i].dtypes == object:
                self.cat_list.append(self.train_x.columns[i])
            else:
                pass
        self.train_x = self.train_x.astype(float)
        self.target_x = self.target_x.astype(float)
        self.sc.fit(self.train_x)
        self.train_x.iloc[:,:] = self.sc.transform(self.train_x.iloc[:,:])
        self.target_x.iloc[:,:] = self.sc.transform(self.target_x.iloc[:,:])
        
    def _num_nan_fill(self):
        self.train_x = self.train_x.fillna(0)
        self.target_x = self.target_x.fillna(0)
    
    def _x_y_split(self):
        self.train_x = self.train.iloc[:,1:]
        self.train_y = self.train.iloc[:,0]
        self.target_x = self.target.iloc[:,1:]
        self.target_y = self.target.iloc[:,0]
    
    def feature_dropping(self, feature_list):
        for i in feature_list:
            self.train_x.drop(i, axis=1, inplace=True)
            self.target_x.drop(i, axis=1, inplace=True)
            if i in self.cat_list:
                self.cat_list.remove(i)
            elif i in self.num_list:
                self.num_list.remove(i)
            else:
                pass

#####################################################################################################

class LGBM_FeatureSelection(object):
    def __init__(self, data, feature_list):
        self.data_x = data.train_x
        self.data_y = data.train_y
        self.cat_list = data.cat_list
        self.params = {
            'boosting_type': 'gbdt',
            'objective':'binary',
            'metric': ['auc'],
            'num_leaves': 200,   
            'learning_rate': 0.01,
            'feature_fraction': 1,
            'bagging_fraction': 1,
            'bagging_freq': 3,
            'verbose': 1
        }
        self.feature_list = feature_list
        self.raw_score = 0
        self.result = [0 for i in range(len(feature_list))]
        
        self.train_x, self.validate_x, self.train_y, self.validate_y = train_test_split(self.data_x, self.data_y, train_size=0.75)
        
        self._raw_score()
    
    def _raw_score(self):
        results={}
        lgbm_train = lgb.Dataset(data=self.train_x, label=self.train_y, categorical_feature=self.cat_list)
        lgbm_validate = lgb.Dataset(data=self.validate_x, label=self.validate_y, categorical_feature=self.cat_list)
        lgbm = lgb.train(self.params,lgbm_train,
                        num_boost_round=1000,
                        valid_sets=(lgbm_validate, lgbm_train),
                        valid_names=('validate','train'),
                        early_stopping_rounds = 30,
                        evals_result= results,
                        verbose_eval=100)
        y_pred = lgbm.predict(self.validate_x, num_iteration=lgbm.best_iteration)
        score = roc_auc_score(self.validate_y,y_pred)
        self.raw_score = score
        
    def feature_regressor(self):
        for i in range(len(self.feature_list)):
            feature = self.feature_list[i]
            print(feature+':')
            remaining_train_x = self.train_x.drop(feature,axis=1)
            remaining_validate_x = self.validate_x.drop(feature,axis=1)
            lgbm_train = lgb.Dataset(data=remaining_train_x, label=self.train_y, categorical_feature=self.cat_list)
            lgbm_validate = lgb.Dataset(data=remaining_validate_x, label=self.validate_y, categorical_feature=self.cat_list)
            try:
                cat_list = self.cat_list + []
                cat_list = cat_list.remove(feature)
                print(cat_list, self.cat_list)
            except:
                pass
            results={}
            lgbm = lgb.train(self.params,lgbm_train,
                            num_boost_round= 1000,
                            valid_sets=(lgbm_validate, lgbm_train),
                            valid_names=('validate','train'),
                            early_stopping_rounds = 30,
                            evals_result= results,
                            verbose_eval=100)
            y_pred = lgbm.predict(remaining_validate_x, num_iteration=lgbm.best_iteration)
            score = roc_auc_score(self.validate_y,y_pred)
            if score - self.raw_score > -1e-4:
                self.result[i]=1

