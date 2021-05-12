# -*- coding: utf-8 -*-
"""
Created on Mon May  3 18:14:02 2021

@author: Arlen
"""

import pandas as pd

def create_features(df, label=None, train = True):
    df['date']=pd.to_datetime(df['date'])
    df['day_of_week'] = df['date'].dt.dayofweek + 1
    df['month'] = df['date'].dt.month
    df['week'] = df['date'].dt.isocalendar().week
    df['quarter'] = df['date'].dt.quarter
    
    if label is not None:
        if train:
            X = df[['date','day_of_week', 'month', 'week', 'quarter',  'wkd']]
            return X
        X = df[['date','day_of_week', 'month', 'week', 'quarter', 'biz_type',  label]]
        return X
    X = df[['date','day_of_week', 'month', 'periods', 'week', 'quarter', 'wkd', 'post_id']]
    return X

def create_features_task1(df, label=None, train = True):
    df['date']=pd.to_datetime(df['date'])
    df['day_of_week'] = df['date'].dt.dayofweek + 1
    df['month'] = df['date'].dt.month
    df['week'] = df['date'].dt.isocalendar().week
    df['quarter'] = df['date'].dt.quarter
    
    if label is not None:
        if train:
            X = df[['date','day_of_week', 'month', 'week', 'quarter',  'wkd']]
            return X
        X = df[['date','day_of_week', 'month', 'week', 'quarter', 'biz_type',  label]]
        return X
    X = df[['date','day_of_week', 'month', 'week', 'quarter', 'wkd', 'post_id']]
    return X