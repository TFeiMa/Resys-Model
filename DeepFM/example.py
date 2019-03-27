#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 17:26:08 2019

@author: ma
"""

import pandas as pd
import numpy as np
from scipy.sparse import lil_matrix

def data_precessing(data_continue, data_cat, quantile=0.999):
    
#    计算分位点
    cat_quantile = data_cat.apply(lambda x: get_cat_quantile(x, q=quantile))
#    计算每个特征值出现的频率
    cat_value_counts = data_cat.apply(lambda x: dict(x.value_counts()))
#    去除低于分位点的取值
    cat_columns = data_cat.columns
    for i in cat_columns:
        data_cat[i] = data_cat.loc[:, i].apply(lambda x: x if isinstance(x, str) and cat_value_counts[i][x] >= \
                                        cat_quantile[i] else float('NaN'))
#    离散特征热编码
    data_cat.fillna('NaN',inplace=True) # 填充为Nan字符，当作固有的特征
    data_cat_dum = pd.DataFrame(index=data_cat.index)
    for i in cat_columns:
        temp = pd.get_dummies(data_cat[i], prefix=int(i))
        data_cat_dum = data_cat_dum.join(temp)

##    连接连续属性和离散属性
#    data = data_continue.join(data_cat_dum)
#    data.fillna(-1, inplace=True) # 将连续特征的缺失值，填充为0
    data_cat_arr = data_cat_dum.values
    data_cat_sp = lil_matrix(data_cat_arr)
    data_cat_index, data_cat_value = data_cat_sp.rows, data_cat_sp.data
    
    data_continue.fillna(0, inplace=True)
    data_continue_rows, data_continue_len = data_continue.shape
    data_continue_index = np.array([list(range(data_continue_len)) for i in range(data_continue_rows)])
    data_continue_value = data_continue.values
    
    data_cat_value = np.array([i for i in data_cat_value])
    data_cat_index = np.array([np.array(i)+data_continue_len for i in data_cat_sp.rows])
    
    
    feat_index = np.concatenate((data_continue_index, data_cat_index), axis=1)
    feat_value = np.concatenate((data_continue_value, data_cat_value), axis=1)
    
    feature_size = data_continue.shape[1] + data_cat_dum.shape[1]
    field_size = data_continue.shape[1] + data_cat.shape[1]
   
    
    return feat_index,feat_value, feature_size, field_size
    
def get_cat_quantile(x, q):
    '''计算x每个值出现的次数的q分位点'''
    cat_counts = x.value_counts()
    return cat_counts.quantile(q)

data_path = '~/Desktop/document/Dataset/criteo_sample/dac_sample.txt'
criteo_data = pd.read_csv(data_path, sep='\t', header=None)
criteo_label = criteo_data.iloc[:,0]
criteo_conti = criteo_data.iloc[:,1:14]
criteo_cat = criteo_data.iloc[:, 14:40]

feat_index, feat_value, feature_size, field_size = data_precessing(criteo_conti[1500:2500], criteo_cat[1500:2500])
labels = criteo_label[1500:2500].values.reshape([-1,1])

from DeepFM import DeepFM
deep_fm = DeepFM(feature_size=feature_size, fild_size=field_size, epochs=100)
deep_fm.train_model(feat_index, feat_value, labels)

