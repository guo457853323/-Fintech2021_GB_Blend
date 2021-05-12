# -*- coding: utf-8 -*-
"""
Created on Tue May  4 20:45:42 2021

@author: ArlenGuo
"""

import pandas as pd
import numpy as np

from zhprepocess.create_features import create_features_task1
from zhprepocess.concat_all import concat_all
from zhFeatureEng.to_data_week import to_train_data
from zhFeatureEng.to_data_week import to_day_test_data

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor

print("\n\n感谢您的审阅，程序运行中，请您等待，预计耗时30s\n")    

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
test_data = pd.read_csv(r"./test_v2_day.csv")

wkd = pd.read_csv(r"./wkd_v1.csv",sep=",",encoding="UTF-8")
wkd = wkd.replace(['WN','SN','NH','SS','WS'],[0,1,2,3,4])
wkd['ORIG_DT']=pd.to_datetime(wkd['ORIG_DT'])
wkd['WKD_TYP_CD'] = wkd['WKD_TYP_CD'].astype(int)
test_data['wkd'] = ''
test_data['date']=pd.to_datetime(test_data['date'])


test_data = concat_all(wkd, test_data, False)


test_data = create_features_task1(test_data)

day_test_data = []
day_test_data.append(to_day_test_data(test_data,A1))
day_test_data.append(to_day_test_data(test_data,A2))
day_test_data.append(to_day_test_data(test_data,A3))
day_test_data.append(to_day_test_data(test_data,A4))
day_test_data.append(to_day_test_data(test_data,A5))
day_test_data.append(to_day_test_data(test_data,A6))
day_test_data.append(to_day_test_data(test_data,A7))
day_test_data.append(to_day_test_data(test_data,A8))
day_test_data.append(to_day_test_data(test_data,A9))
day_test_data.append(to_day_test_data(test_data,A10))
day_test_data.append(to_day_test_data(test_data,A11))
day_test_data.append(to_day_test_data(test_data,A12))
day_test_data.append(to_day_test_data(test_data,A13))
day_test_data.append(to_day_test_data(test_data,B1))

# =============================================================================
# train_data
# =============================================================================
day_data = []
day_data.append(to_train_data(A1))
day_data.append(to_train_data(A2))
day_data.append(to_train_data(A3))
day_data.append(to_train_data(A4))
day_data.append(to_train_data(A5))
day_data.append(to_train_data(A6))
day_data.append(to_train_data(A7))
day_data.append(to_train_data(A8))
day_data.append(to_train_data(A9))
day_data.append(to_train_data(A10))
day_data.append(to_train_data(A11))
day_data.append(to_train_data(A12))
day_data.append(to_train_data(A13))
day_data.append(to_train_data(B1))


# =============================================================================
# Training&Predicting_stage
# =============================================================================

A_amount = []
B_amount = []

for i in range(len(day_data)):
    print("正在进行第 "+ str(i + 1) +"个业务的模型训练与预测")
    reg1 = XGBRegressor(tree_method='gpu_hist', gpu_id=0)
    reg2 = LGBMRegressor()
    reg3 = AdaBoostRegressor()
    reg4 = GradientBoostingRegressor()
    X_train = day_data[i][0]
    X_train['wkd'] = X_train['wkd'].astype(int)
    X_train = X_train.astype(float)
    y_train = np.array(day_data[i][1])
    y_train = y_train.ravel()
    X_test = day_test_data[i]
    X_test = X_test.astype(float)
    
    
    reg1.fit(X_train, y_train)
    reg2.fit(X_train, y_train)
    reg3.fit(X_train, y_train)
    reg4.fit(X_train, y_train)
    
    reg1_result = reg1.predict(X_test)
    reg2_result = reg2.predict(X_test)
    reg3_result = reg3.predict(X_test)
    reg4_result = reg4.predict(X_test)
    
    forecast_data = 0.4 * reg1_result + 0.2 * reg2_result + 0.2 * reg3_result + 0.2 * reg4_result
    forecast_data = forecast_data.astype(int)
    if i <= 12:
        A_amount.append(forecast_data)
    else:
        B_amount.append(forecast_data)

A_result = A_amount[0]
B_result = B_amount[0]
final_result = []

for i in range(len(A_amount) - 1):
    A_result = np.array(A_amount[i]) + np.array(A_result)

A_result = (A_result * 1.14).astype(int)
B_result = (B_result * 1.12).astype(int)

for i in range(62):
    if i % 2 == 0:
        final_result.append(A_result[i])
    else:
        final_result.append(B_result[i])




final_result = np.array(final_result)
for i in range(len(final_result)):
    if final_result[i] < 1000:
        final_result[i] = 0
   
print("\n运行完成，预测结果保存至'final_result'变量中\n请注意：由于GBDT与AdaBoost列采样等随机性每次运行结果稍有不同，但并不影响整体准确率\n")
# =============================================================================
# result_save
# =============================================================================
         
# 需要手动操作保存至csv中，再运行下列代码，结果保存在result文件夹中
print('\n最终结果需要手动操作保存至test***.csv中，再运行底端注释代码保存为txt文件，txt结果保存在result文件夹中')
# result = pd.read_csv(r"./test_v2_day.csv")
# result.columns = ["date", "post_id", "amount"]
# result.to_csv(r"./result/period_result_blend_10week_1.14A.txt", index = 0)