# -*- coding: utf-8 -*-
"""
Created on Tue May  4 20:45:42 2021

@author: ArlenGuo
"""


import pandas as pd
import numpy as np

from zhprepocess.create_features import create_features

from zhprepocess.concat_all import concat_all
from zhFeatureEng.to_data import to_periods_train_data
from zhFeatureEng.to_data import to_periods_test_data


from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


print("\n感谢您的审阅，程序运行中，请您等待，预计耗时30s\n")   

A1 =   pd.read_csv(r"./pocessed_data/A1.csv")
A2 =   pd.read_csv(r"./pocessed_data/A2.csv")
A3 =   pd.read_csv(r"./pocessed_data/A3.csv")
A4 =   pd.read_csv(r"./pocessed_data/A4.csv")
A5 =   pd.read_csv(r"./pocessed_data/A5.csv")
A6 =   pd.read_csv(r"./pocessed_data/A6.csv")
A7 =   pd.read_csv(r"./pocessed_data/A7.csv")
A8 =   pd.read_csv(r"./pocessed_data/A8.csv")
A9 =   pd.read_csv(r"./pocessed_data/A9.csv")
A10 =   pd.read_csv(r"./pocessed_data/A10.csv")
A11 =   pd.read_csv(r"./pocessed_data/A11.csv")
A12 =   pd.read_csv(r"./pocessed_data/A12.csv")
A13 =   pd.read_csv(r"./pocessed_data/A13.csv")
B1 =   pd.read_csv(r"./pocessed_data/B1.csv")

# =============================================================================
# test_data
# =============================================================================

test_data = pd.read_csv(r"./test_v2_periods.csv") 

wkd = pd.read_csv(r"./wkd_v1.csv",sep=",",encoding="UTF-8")
wkd = wkd.replace(['WN','SN','NH','SS','WS'],[0,1,2,3,4])
wkd['ORIG_DT']=pd.to_datetime(wkd['ORIG_DT'])
wkd['WKD_TYP_CD'] = wkd['WKD_TYP_CD'].astype(int)
test_data['wkd'] = ''
test_data['date']=pd.to_datetime(test_data['date'])

test_data = concat_all(wkd, test_data)


test_data = create_features(test_data)

periods_test_data = []
periods_test_data.append(to_periods_test_data(test_data,A1))
periods_test_data.append(to_periods_test_data(test_data,A2))
periods_test_data.append(to_periods_test_data(test_data,A3))
periods_test_data.append(to_periods_test_data(test_data,A4))
periods_test_data.append(to_periods_test_data(test_data,A5))
periods_test_data.append(to_periods_test_data(test_data,A6))
periods_test_data.append(to_periods_test_data(test_data,A7))
periods_test_data.append(to_periods_test_data(test_data,A8))
periods_test_data.append(to_periods_test_data(test_data,A9))
periods_test_data.append(to_periods_test_data(test_data,A10))
periods_test_data.append(to_periods_test_data(test_data,A11))
periods_test_data.append(to_periods_test_data(test_data,A12))
periods_test_data.append(to_periods_test_data(test_data,A13))
periods_test_data.append(to_periods_test_data(test_data,B1))

# =============================================================================
# train_data
# =============================================================================
periods_data = []
periods_data.append(to_periods_train_data(A1))
periods_data.append(to_periods_train_data(A2))
periods_data.append(to_periods_train_data(A3))
periods_data.append(to_periods_train_data(A4))
periods_data.append(to_periods_train_data(A5))
periods_data.append(to_periods_train_data(A6))
periods_data.append(to_periods_train_data(A7))
periods_data.append(to_periods_train_data(A8))
periods_data.append(to_periods_train_data(A9))
periods_data.append(to_periods_train_data(A10))
periods_data.append(to_periods_train_data(A11))
periods_data.append(to_periods_train_data(A12))
periods_data.append(to_periods_train_data(A13))
periods_data.append(to_periods_train_data(B1))


# =============================================================================
# Training&Predicting_stage
# =============================================================================
A_periods_amount = []
B_periods_amount = []


for i in range(len(periods_data)):
    print("正在进行第 "+ str(i + 1) +"个业务的模型训练与预测")
    
    reg1 = XGBRegressor(tree_method='gpu_hist', gpu_id=0)
    reg2 = LGBMRegressor()

    X_train = periods_data[i][0]
    X_train = X_train.astype(float)
    y_train = np.array(periods_data[i][1])
    y_train = y_train.ravel()
    X_test = periods_test_data[i]
    X_test = X_test.astype(float)
    
    reg1.fit(X_train, y_train)
    reg2.fit(X_train, y_train)

    
    reg1_result = reg1.predict(X_test)
    reg2_result = reg2.predict(X_test)

    
    forecast_data = 0.5 * reg1_result + 0.5 * reg2_result

    forecast_data = forecast_data.astype(int)

    if i <= 12:
        A_periods_amount.append(forecast_data)
    else:
        B_periods_amount.append(forecast_data)

A_result = A_periods_amount[0]
B_result = B_periods_amount[0]   
final_result = []

for i in range(len(A_periods_amount) - 1):
    A_result = np.array(A_periods_amount[i]) + np.array(A_result)

for i in range(62):
    if i % 2 == 0:
        final_result.append(A_result[i * 48 : i * 48 + 48] * 1.14)
    else:
        final_result.append(B_result[i * 48 : i * 48 + 48] )

a = final_result[0]
for i in range(61):
    a = np.hstack((a, final_result[i + 1]))
    
for i in range(len(a)):
    if a[i] < 0:
        a[i] = 0

for i in range(60):
    a[0 + i*48 : 17 + i*48] = 0
    a[38 + i*48 : 48 + i*48] = 0
a = a.astype(int)

result = test_data.loc[:,['day_of_week','post_id','periods']]
result.loc[:,'amount'] = pd.DataFrame(a).loc[:, 0]
b = result[((result['day_of_week'] == 6) | (result['day_of_week'] == 7)) & (result['post_id'] == 'B')| (result['periods'] < 17) | (result['periods'] >= 37)].index
result.loc[b,'amount'] = 0 
result = np.array(result['amount'])
final_result = result
print("\n运行完成，预测结果保存至'final_result'变量中")

# =============================================================================
# result_save
# =============================================================================
# 需要手动操作保存至csv中，再运行下列代码，结果保存在result文件夹中
print('\n最终结果需要手动操作保存至test***.csv中，再运行底端注释代码保存为txt文件，txt结果保存在result文件夹中')
# b = pd.read_csv('test_v2_periods.csv')
# b['periods'] = b['periods'].astype(int)
# b.to_csv(r"./result/period_result_blend_10week_1.14A_tenMean_moreFeatures.txt", index = 0)