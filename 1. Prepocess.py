# -*- coding: utf-8 -*-
"""
Created on Mon May  3 18:20:23 2021

@author: Arlen
"""

import pandas as pd
import numpy as np

from zhprepocess.concat_all import concat_all
from zhprepocess.split_type import split_type
# =============================================================================
# 1. 合并wkd与train_data表
# =============================================================================

print('这一步很慢，建议直接运行task1与task2，预处理后数据已经在压缩包中给出')

# 读取数据
train_data = pd.read_csv(r"./train_v2.csv")
wkd = pd.read_csv(r"./wkd_v1.csv",sep=",",encoding="UTF-8")

# date列转换为datetime格式
train_data['date']=pd.to_datetime(train_data['date'])

# 节假日类型数字化表示
wkd = wkd.replace(['WN','SN','NH','SS','WS'],[0,1,2,3,4])
wkd['ORIG_DT']=pd.to_datetime(wkd['ORIG_DT'])
train_data['wkd'] = ''

# 合并表
train_data = concat_all(wkd, train_data) 



# =============================================================================
# 2. 分割不同业务数据
# =============================================================================

type_data = split_type(train_data)


# 之后跳转至task1与task2
