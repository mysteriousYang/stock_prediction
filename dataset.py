# -*- coding:utf-8 -*-
import os
import sys
import torch
import sklearn
import json
import pandas as pd
import numpy as np
from logger import Enable_Logger
from file_utility import check_paths
from torch.utils.data import DataLoader,Dataset
from sklearn.preprocessing import normalize

sparse_cols = [
    "timestamp",
    "volume",
    "amount",
    "market_capital",
    "balance",
    "hold_volume_cn",
    "net_volume_cn",
]

dense_cols = [
    "open",
    "high",
    "low",
    "chg",
    "percent",
    "turnoverrate",
    "pe",
    "pb",
    "ps",
    "pcf",
    "hold_ratio_cn",
]

class Single_Stock_Dataset(Dataset):
    def __init__(self, dense_X, sparse_X, Y,):
        super().__init__()
        self.dense_X = dense_X
        self.sparse_X = sparse_X
        self.Y = Y

    def __len__(self):
        return len(self.Y)
    
    def __getitem__(self, index):
        return self.dense_X[index], self.sparse_X[index], self.Y[index]
    

class Multi_Stocks_Dataset(Dataset):
    def __init__(self, dense_X, sparse_X, Y,):
        super().__init__()
        self.dense_X = dense_X
        self.sparse_X = sparse_X
        self.Y = Y

    def __len__(self):
        return len(self.Y)
    
    def __getitem__(self, index):
        return self.dense_X[index], self.sparse_X[index], self.Y[index]
    
def build_single_dataset(symbol:str):
    '''
    为symbol对应的股票创建数据集, 使用day, week, month字段
    '''

    # 读取csv并转化为ndarray
    # using_periods = ["day","week","month"] 
    sparse_X = np.array([],dtype=np.int64).reshape(0,len(sparse_cols))
    dense_X = np.array([],dtype=np.float64).reshape(0,len(dense_cols))
    label_Y = np.array([],dtype=np.float64)

    for period in ["day","week","month"]:
        csv_name = symbol + "_" + period + ".csv"
        csv_name = os.path.join(".\\data\\stocks",symbol,csv_name)
        
        df = pd.read_csv(csv_name,encoding="utf-8")
        # 缺失值处理:填充为0
        # df = df.ffill()
        df = df.fillna(0)

        #读取离散数据
        sparse_sample = np.array(df[sparse_cols],dtype=np.int64)

        #读取连续数据
        dense_sample = np.array(df[dense_cols],dtype=np.float64)

        #读取收盘价
        label_sample = np.array(df["close"],np.float64)

        sparse_X = np.vstack((sparse_X, sparse_sample))
        dense_X = np.vstack((dense_X, dense_sample))
        label_Y = np.hstack((label_Y, label_sample))

    # print(sparse_X)
    # print(dense_X)
    # print(len(label_Y))


    # 数据归一化
    # dense_X[np.isnan(dense_X)] = 0
    dense_X = normalize(dense_X,"max",axis=0)
    # print(dense_X)

    return Single_Stock_Dataset(dense_X, sparse_X, label_Y)


if __name__ == "__main__":
    check_paths()
    Enable_Logger()
    
    build_single_dataset("SH600221")
    pass