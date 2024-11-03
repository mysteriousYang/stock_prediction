# -*- coding:utf-8 -*-
import os
import sys
import torch
# import sklearn
import json
import pandas as pd
import numpy as np
from logger import Enable_Logger
from file_utility import check_paths
from torch.utils.data import DataLoader,Dataset
from sklearn.preprocessing import normalize

sparse_cols = [
    # "timestamp",
    # "volume",
    # "amount",
    # "market_capital",
    # "balance",
    # "hold_volume_cn",
    # "net_volume_cn",
]

dense_cols = [
    # "timestamp",
    "volume",
    "amount",
    "market_capital",
    "balance",
    "hold_volume_cn",
    "net_volume_cn",
    # "open",
    # "high",
    # "low",
    "chg",
    "percent",
    # "turnoverrate",
    # "pe",
    "pb",
    "ps",
    "pcf",
    "hold_ratio_cn",
]

time_cols = [
    "timestamp",
    "open",
    "high",
    "low",
    "turnoverrate",
    "pe"
]

dense_input_size = len(dense_cols)
sparse_input_size = len(sparse_cols)
time_series_input_size = len(time_cols)


class Last_Day_Sample():
    def __init__(self, sparse, dense, time_series):
        self.sparse = sparse
        self.dense = dense
        self.time_series = time_series

        # print(self.sparse,self.dense,self,time_series)


class Single_Stock_Dataset(Dataset):
    def __init__(self, 
                 dense_X, 
                 sparse_X, 
                 time_X, 
                 Y, 
                 last_day_sample:Last_Day_Sample,
                 symbol:str,
                 name:str):
        super().__init__()
        self.dense_X = dense_X
        self.sparse_X = sparse_X
        self.time_X = time_X
        self.Y = Y
        self.last_day = last_day_sample
        self.symbol = symbol
        self.name = name

    def __len__(self):
        return len(self.Y)
    
    def __getitem__(self, index):
        return self.dense_X[index], self.sparse_X[index], self.time_X[index], self.Y[index]
    
    def last_dense(self, device):
        # return torch.tensor(self.last_day.dense,dtype=torch.float64).unsqueeze(0).to(device)
        return self.last_day.dense.unsqueeze(0)
    
    def last_sparse(self, device):
        # return torch.tensor(self.last_day.sparse,dtype=torch.float64).unsqueeze(0).to(device)
        return self.last_day.sparse.unsqueeze(0)
    
    def last_time_series(self, device):
        # return torch.tensor(self.last_day.time_series,dtype=torch.float64).unsqueeze(0).to(device)
        return self.last_day.time_series.unsqueeze(0)
    

class Multi_Stocks_Dataset(Dataset):
    def __init__(self, dense_X, sparse_X, time_X, Y,):
        super().__init__()
        self.dense_X = dense_X
        self.sparse_X = sparse_X
        self.time_X = time_X
        self.Y = Y

    def __len__(self):
        return len(self.Y)
    
    def __getitem__(self, index):
        return self.dense_X[index], self.sparse_X[index], self.time_X[index], self.Y[index]
    
def build_single_dataset(symbol:str,device:str):
    '''
    为symbol对应的股票创建数据集, 使用day, week, month字段
    '''

    # 读取csv并转化为ndarray
    # using_periods = ["day","week","month"] 
    sparse_X = np.array([],dtype=np.int64).reshape(0,len(sparse_cols))
    dense_X = np.array([],dtype=np.float64).reshape(0,len(dense_cols))
    time_X = np.array([],dtype=np.float64).reshape(0,len(time_cols))

    # 因为这里一次预测15天, 所以一个Y含有15个元素
    label_Y = np.array([],dtype=np.float64).reshape(0, 15)

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

        # 读取时间序列数据
        time_sample = np.array(df[time_cols],dtype=np.float64)

        #读取收盘价
        close_price = np.array(df["close"],np.float64)
        if(period == "day"):
            # 将15天的数据合为一个Y label
            final_index = len(df) - 15
            label_sample = np.array([close_price[i:i+15] for i in range(final_index)], dtype=np.float64)

            last_day = Last_Day_Sample(sparse_sample[-1],dense_sample[-1],time_sample[-1])


        elif(period == "week"):
            # 将2周的数据合为一个Y lebel
            final_index = len(df) - 1
            # print(len(df),final_index)
            label_sample = np.array([],dtype=np.float64).reshape(0, 15)
            for i in range(final_index):
                f7 = np.full(7,close_price[i], dtype=np.float64)
                b7 = np.full(7,close_price[i+1], dtype=np.float64)
                avg = (close_price[i]+close_price[i+1])/2
                label_sample = np.vstack((label_sample, np.hstack((f7,avg,b7), dtype=np.float64)))

        elif(period == "month"):
            label_sample = np.array([np.full(15,value) for value in close_price], dtype=np.float64)

        # print(len(dense_sample),len(time_sample),len(label_sample))

        sparse_X = np.vstack((sparse_X, sparse_sample[:len(label_sample)]))
        dense_X = np.vstack((dense_X, dense_sample[:len(label_sample)]))
        time_X = np.vstack((time_X, time_sample[:len(label_sample)]))
        label_Y = np.vstack((label_Y, label_sample))

    sparse_X = np.vstack((sparse_X, last_day.sparse))
    dense_X = np.vstack((dense_X, last_day.dense))
    time_X = np.vstack((time_X, last_day.time_series))

    # print(len(sparse_X))
    # print(len(dense_X))
    # print(len(time_X))
    # print(len(label_Y))


    # 数据归一化
    # dense_X[np.isnan(dense_X)] = 0
    dense_X = normalize(dense_X,"max",axis=0)
    time_X = normalize(time_X,"max",axis=0)
    # print(dense_X)
    # print(time_X)

    # 可能需要放到gpu上
    sparse_X = torch.tensor(sparse_X[:-1],dtype=torch.int64).to(device)
    dense_X = torch.tensor(dense_X[:-1],dtype=torch.float64).to(device)
    time_X = torch.tensor(time_X[:-1],dtype=torch.float64).to(device)
    label_Y = torch.tensor(label_Y,dtype=torch.float64).to(device)

    last_day = Last_Day_Sample(sparse_X[-1],dense_X[-1],time_X[-1])

    # 获得股票名与编号
    with open(os.path.join(f".\\data\\stocks\\{symbol}",f"{symbol}_info.json"),"r",encoding="utf-8") as fp:
        json_data = json.load(fp)

    return Single_Stock_Dataset(dense_X, sparse_X, time_X, label_Y, last_day,
                                symbol, json_data["name"])


if __name__ == "__main__":
    check_paths()
    Enable_Logger()
    
    build_single_dataset("SH600221","cpu")
    pass