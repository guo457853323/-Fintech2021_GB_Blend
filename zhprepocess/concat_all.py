# -*- coding: utf-8 -*-
"""
Created on Mon May  3 22:02:02 2021

@author: Arlen
"""


# =============================================================================
# 合并节假日与原矩阵
# =============================================================================
def concat_all(wkd, train_data, train=True):
    train_data_len = len(train_data)
    if train:
        for i in range(len(wkd)):
            w_date = wkd.loc[i, 'ORIG_DT']
            wkd_num = wkd.loc[i, 'WKD_TYP_CD']
            j = 0
            while j < train_data_len:
                t_date = train_data.loc[j, 'date']
                if t_date == w_date:
                    train_data.loc[j:j + 47, 'wkd'] = wkd_num
                j += 48
            # print(i)
    else:
        for i in range(len(wkd)):
            w_date = wkd.loc[i, 'ORIG_DT']
            wkd_num = wkd.loc[i, 'WKD_TYP_CD']
            j = 0
            while j < train_data_len:
                t_date = train_data.loc[j, 'date']
                if t_date == w_date:
                    train_data.loc[j:j + 2, 'wkd'] = wkd_num
                j += 2
            # print(i)
    return train_data

