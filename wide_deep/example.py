import pandas as pd
import numpy as np
from scipy.sparse import lil_matrix

def data_precessing(data_cat, quantile=0.999):
    
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
    
    data_cat_value = np.array([i for i in data_cat_value])
    data_cat_index = np.array([np.array(i) for i in data_cat_sp.rows])
    
    feature_size = data_cat_dum.shape[1]
    field_size =  data_cat.shape[1]
    
    return data_cat_index,data_cat_value, feature_size, field_size, data_cat_dum
    
def get_cat_quantile(x, q):
    '''计算x每个值出现的次数的q分位点'''
    cat_counts = x.value_counts()
    return cat_counts.quantile(q)

data_path = '~/Desktop/document/Dataset/criteo_sample/dac_sample.txt'
criteo_data = pd.read_csv(data_path, sep='\t', header=None)[1500:2500]
criteo_label = criteo_data.iloc[:,0]
criteo_conti = criteo_data.iloc[:,1:14].fillna(0)
criteo_cat = criteo_data.iloc[:, 14:40]

feat_index, feat_value, feature_size, field_size, data_cat_dum = data_precessing(criteo_cat)
x_wide = np.concatenate((criteo_conti.values, data_cat_dum), axis=1)
wide_size = x_wide.shape[1]
numeric_size = criteo_conti.shape[1]
labels = criteo_label.values.reshape([-1,1])

from wide_deep import Wide_Deep
wide_deep = Wide_Deep(wide_size=wide_size, field_size=field_size, sparse_size=feature_size, 
    numeric_size=numeric_size, epochs=100, learning_rate=0.005)
wide_deep.train_model(x_wide, criteo_conti.values, feat_index, feat_value, labels)

