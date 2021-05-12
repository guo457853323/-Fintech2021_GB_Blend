# -*- coding: utf-8 -*-
"""
Created on Tue May  4 19:18:28 2021

@author: Arlen
"""

import pandas as pd

def to_train_data(data, plus=False):
    data['date']=pd.to_datetime(data['date'])
    A_train = data
    A_train['day_of_week'] = A_train['date'].dt.weekday + 1
    A_train['month'] = A_train['date'].dt.month
    A_train['week'] = A_train['date'].dt.isocalendar().week
    A_train['quarter'] = A_train['date'].dt.quarter
    A_train['day'] = A_train['date'].dt.day

    # # 添加周中与非周中信息
    # A_train['week_start_end'] = ''
    # start_index = A_train[(A_train['day_of_week'] == 1) | (A_train['day_of_week'] == 5)].index
    # A_train.loc[start_index,'week_start_end'] = int(2) 
    # end_index = A_train[(A_train['day_of_week'] == 2) | (A_train['day_of_week'] == 3) | (A_train['day_of_week'] == 4)].index
    # A_train.loc[end_index,'week_start_end'] = int(1)
    # weekend_index = A_train[(A_train['day_of_week'] == 6) | (A_train['day_of_week'] == 7)].index
    # A_train.loc[weekend_index,'week_start_end'] = int(1)
    # A_train['week_start_end'] = A_train['week_start_end'].astype(int)
    # # 添加月中、月初、月末信息
    # A_train['month_start_end'] = ''
    # start_index = A_train[(A_train['day'] <=10)].index
    # A_train.loc[start_index,'month_start_end'] = int(1) 
    # end_index = A_train[(A_train['day'] > 20)].index
    # A_train.loc[end_index,'month_start_end'] = int(2)
    # mid_index = A_train[(A_train['day'] <= 20) & (A_train['day'] > 10)].index
    # A_train.loc[mid_index,'month_start_end'] = int(3) 
    # A_train['month_start_end'] = A_train['month_start_end'].astype(int)
    
    start = 47760
    end = 51120
    gap = 336
    for i in range(30):
     # 默认提取前6~11周特征
        for j in range(10):
            A_train.loc[start + i * gap : start + (i + 1) * gap, ('last_'+str(j + 1)+'_week_mean')] = A_train.loc[start  + (i - j - 6) * gap : start + (i-j-5) * gap, ( 'amount')].mean()
            A_train.loc[start + i * gap : start + (i + 1) * gap, ('last_'+str(j + 1)+'_week_std')] = A_train.loc[start  + (i - j - 6) * gap : start + (i-j-5) * gap, ( 'amount')].std(ddof=0)
            A_train.loc[start + i * gap : start + (i + 1) * gap, ('last_'+str(j + 1)+'_week_max')] = A_train.loc[start  + (i - j - 6) * gap : start + (i-j-5) * gap, ( 'amount')].max()
            A_train.loc[start + i * gap : start + (i + 1) * gap, ('last_'+str(j + 1)+'_week_skew')] = A_train.loc[start  + (i - j - 6) * gap : start + (i-j-5) * gap, ( 'amount')].skew()   
    A_train = A_train[start:end]
    A_train=A_train.groupby(A_train['date']).apply(concat_func1).reset_index()

    A_train.columns = ['date',
                       'day',
                        'day_of_week',
                        'month',
                        'wkd', 
                        'week', 
                        'quarter',
                        'amount',
                        'last_1_week_mean',
                        'last_1_week_std',
                        'last_2_week_mean', 
                        'last_2_week_std',
                        'last_3_week_mean',
                        'last_3_week_std',
                        'last_4_week_mean',
                        'last_4_week_std',
                        'last_5_week_mean',
                        'last_5_week_std',
                        'last_6_week_mean',
                        'last_6_week_std',
                        'last_7_week_mean',
                        'last_7_week_std',
                        # 'last_8_week_mean',
                        # 'last_8_week_std', 
                        # 'last_9_week_mean',
                        # 'last_9_week_std', 
                        # 'last_10_week_mean',
                        # 'last_10_week_std', 
                        # 'week_start_end',
                        # 'month_start_end',
                        
                        # 'last_1_week_max',
                        # 'last_2_week_max',
                        # 'last_3_week_max',
                        # 'last_4_week_skew',
                        # 'last_5_week_skew',
                        # 'last_6_week_skew',
                        # 'last_7_week_skew',
                        ]
    X_train = pd.DataFrame(A_train, columns = ['day',
                                               'day_of_week',
                                               'month',
                                               'wkd', 
                                               'week', 
                                               'quarter',
                                                'last_1_week_mean',
                                                'last_1_week_std',
                                                'last_2_week_mean', 
                                                'last_2_week_std',
                                                'last_3_week_mean',
                                                'last_3_week_std',
                                               
                                                'last_4_week_mean',
                                                'last_4_week_std',
                                                 'last_5_week_mean',
                                                 'last_5_week_std',
                                                 'last_6_week_mean',
                                                 'last_6_week_std',
                                                 'last_7_week_mean',
                                                 'last_7_week_std',  
                                               #  'last_8_week_mean',
                                               #  'last_8_week_std', 
                                               #  'last_9_week_mean',
                                               #  'last_9_week_std', 
                                               #  'last_10_week_mean',
                                               #  'last_10_week_std', 
                                                # 'week_start_end',
                                                # 'month_start_end',
                                                # 'last_1_week_max',
                                                # 'last_2_week_max',
                                                # 'last_3_week_max',
                                                # 'last_1_week_skew',
                                                # 'last_2_week_skew',
                                                # 'last_3_week_skew',
                                                # 'last_4_week_skew',
                                                # 'last_5_week_skew',
                                                # 'last_6_week_skew',
                                                # 'last_7_week_skew',
                                               ])
   
    y_train = pd.DataFrame(A_train, columns = ['amount'])
    data_list = []
    data_list.append(X_train)
    data_list.append(y_train)

    
    return data_list
    
    




def to_day_test_data(data, last):
    data['date']=pd.to_datetime(data['date'])
    # A=data.groupby(data['date']).apply(concat_func1).reset_index()
    A_test = data
    A_test['day_of_week'] = A_test['date'].dt.weekday + 1
    A_test['month'] = A_test['date'].dt.month
    A_test['week'] = A_test['date'].dt.isocalendar().week
    A_test['quarter'] = A_test['date'].dt.quarter
    A_test['day'] = A_test['date'].dt.day

    # A_test['week_start_end'] = ''
    # start_index = A_test[(A_test['day_of_week'] == 1) | (A_test['day_of_week'] == 5)].index
    # A_test.loc[start_index,'week_start_end'] = int(2) 
    # end_index = A_test[(A_test['day_of_week'] == 2) | (A_test['day_of_week'] == 3) | (A_test['day_of_week'] == 4)].index
    # A_test.loc[end_index,'week_start_end'] = int(1)
    # weekend_index = A_test[(A_test['day_of_week'] == 6) | (A_test['day_of_week'] == 7)].index
    # A_test.loc[weekend_index,'week_start_end'] = int(1)
    # A_test['week_start_end'] = A_test['week_start_end'].astype(int)
    # # 添加月中、月初、月末信息
    # A_test['month_start_end'] = ''
    # start_index = A_test[(A_test['day'] <=10)].index
    # A_test.loc[start_index,'month_start_end'] = int(1) 
    # end_index = A_test[(A_test['day'] > 20)].index
    # A_test.loc[end_index,'month_start_end'] = int(2)
    # mid_index = A_test[(A_test['day'] <= 20) & (A_test['day'] > 10)].index
    # A_test.loc[mid_index,'month_start_end'] = int(3) 
    # A_test['month_start_end'] = A_test['month_start_end'].astype(int)

    end = 51120
    gap = 336
    for i in range(30):
        for j in range(10):
            A_test.loc[61 - 14*(i + 1) : 61 - 14*i, ('last_'+str(j + 1)+'_week_mean')] = last.loc[end - (j + 6 + i)*gap : end - (j + 5 + i)*gap, ( 'amount')].mean()
            A_test.loc[61 - 14*(i + 1) : 61 - 14*i, ('last_'+str(j + 1)+'_week_std')] = last.loc[end - (j + 6 + i)*gap : end - (j + 5 + i)*gap, ( 'amount')].std(ddof=0)
            A_test.loc[61 - 14*(i + 1) : 61 - 14*i, ('last_'+str(j + 1)+'_week_max')] = last.loc[end - (j + 6 + i)*gap : end - (j + 5 + i)*gap, ( 'amount')].max()
    X_test = pd.DataFrame(A_test, columns =  ['day',
                                               'day_of_week',
                                               'month',
                                               'wkd', 
                                               'week', 
                                               'quarter',
                                                'last_1_week_mean',
                                                'last_1_week_std',
                                                'last_2_week_mean', 
                                                'last_2_week_std',
                                                'last_3_week_mean',
                                                'last_3_week_std',
                                                'last_4_week_mean',
                                                'last_4_week_std',
                                                 'last_5_week_mean',
                                                 'last_5_week_std',
                                                 'last_6_week_mean',
                                                 'last_6_week_std',
                                                 'last_7_week_mean',
                                                 'last_7_week_std',
                                               #  'last_8_week_mean',
                                               #  'last_8_week_std', 
                                               #  'last_9_week_mean',
                                               #  'last_9_week_std', 
                                               #  'last_10_week_mean',
                                               #  'last_10_week_std', 
                                                # 'week_start_end',
                                                # 'month_start_end',
                                                # 'last_1_week_max',
                                                # 'last_2_week_max',
                                                # 'last_3_week_max',
                                               ])
    return X_test
    


def concat_func1(x):
    return pd.Series([
        x['day'].mean(),
        x['day_of_week'].mean(),
        x['month'].mean(),
        x['wkd'].mean(),
        x['week'].mean(),
        x['quarter'].mean(),
        x['amount'].sum(),
        x['last_1_week_mean'].mean(),
        x['last_1_week_std'].mean(),
        x['last_2_week_mean'].mean(),
        x['last_2_week_std'].mean(),
        x['last_3_week_mean'].mean(),
        x['last_3_week_std'].mean(),
        x['last_4_week_mean'].mean(),
        x['last_4_week_std'].mean(),
        x['last_5_week_mean'].mean(),
        x['last_5_week_std'].mean(),
        x['last_6_week_mean'].mean(),
        x['last_6_week_std'].mean(),
        x['last_7_week_mean'].mean(),
        x['last_7_week_std'].mean(),
        # x['last_8_week_mean'].mean(),
        # x['last_8_week_std'].mean(),
        # x['last_9_week_mean'].mean(),
        # x['last_9_week_std'].mean(),
        # x['last_10_week_mean'].mean(),
        # x['last_10_week_std'].mean(),
        # x['week_start_end'].mean(),
        # x['month_start_end'].mean(),
        # x['last_1_week_max'].mean(),
        # x['last_2_week_max'].mean(),
        # x['last_3_week_max'].mean(),
        # x['last_1_week_skew'].mean(),
        # x['last_2_week_skew'].mean(),
        # x['last_3_week_skew'].mean(),
        # x['last_4_week_skew'].mean(),
        # x['last_5_week_skew'].mean(),
        # x['last_6_week_skew'].mean(),
        # x['last_7_week_skew'].mean(),
        
        ]
    )





