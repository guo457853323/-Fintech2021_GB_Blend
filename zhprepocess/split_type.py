# -*- coding: utf-8 -*-
"""
Created on Mon May  3 23:45:24 2021

@author: Arlen
"""

# =============================================================================
# 将每种业务数据分离
# =============================================================================
import pandas as pd
def split_type(train_data):
    A1 = train_data[(train_data['biz_type'] == 'A1')]
    A2 = train_data[(train_data['biz_type'] == 'A2')]
    A3 = train_data[(train_data['biz_type'] == 'A3')]
    A4 = train_data[(train_data['biz_type'] == 'A4')]
    A5 = train_data[(train_data['biz_type'] == 'A5')]
    A6 = train_data[(train_data['biz_type'] == 'A6')]
    A7 = train_data[(train_data['biz_type'] == 'A7')]
    A8 = train_data[(train_data['biz_type'] == 'A8')]
    A9 = train_data[(train_data['biz_type'] == 'A9')]
    A10 = train_data[(train_data['biz_type'] == 'A10')]
    A11 = train_data[(train_data['biz_type'] == 'A11')]
    A12 = train_data[(train_data['biz_type'] == 'A12')]
    A13 = train_data[(train_data['biz_type'] == 'A13')]
    B1 = train_data[(train_data['biz_type'] == 'B1')]
    
    A1.to_csv(r"./pocessed_data/A1.csv", index = 0)
    A2.to_csv(r"./pocessed_data/A2.csv", index = 0)
    A3.to_csv(r"./pocessed_data/A3.csv", index = 0)
    A4.to_csv(r"./pocessed_data/A4.csv", index = 0)
    A5.to_csv(r"./pocessed_data/A5.csv", index = 0)
    A6.to_csv(r"./pocessed_data/A6.csv", index = 0)
    A7.to_csv(r"./pocessed_data/A7.csv", index = 0)
    A8.to_csv(r"./pocessed_data/A8.csv", index = 0)
    A9.to_csv(r"./pocessed_data/A9.csv", index = 0)
    A10.to_csv(r"./pocessed_data/A10.csv", index = 0)
    A11.to_csv(r"./pocessed_data/A11.csv", index = 0)
    A12.to_csv(r"./pocessed_data/A12.csv", index = 0)
    A13.to_csv(r"./pocessed_data/A13.csv", index = 0)
    B1.to_csv(r"./pocessed_data/B1.csv", index = 0)
    return 