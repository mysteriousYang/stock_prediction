# -*- coding:utf-8 -*-
# 该文件用于创建训练集供神经网络训练使用
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

# 分类数据列名, 目前暂时没有用上
# 以后可填入例如产业类型,上市交易所等信息
sparse_cols = [
    # "timestamp",
    # "volume",
    # "amount",
    # "market_capital",
    # "balance",
    # "hold_volume_cn",
    # "net_volume_cn",
]

# 连续数据列名
dense_cols = [
    "volume",
    "amount",
    "market_capital",
    "balance",
    "hold_volume_cn",
    "net_volume_cn",
    "chg",
    "percent",
    "pb",
    "ps",
    "pcf",
    "hold_ratio_cn",
]

# 时间序列数据, 用于LSTM网络输入
time_cols = [
    "timestamp", # 时间戳
    "open", # 开盘价
    "high", # 最高价
    "low", # 最低价
    "turnoverrate", # 换手率
    "pe" # 市盈率
]

# 这三个变量用于构建神经网络结构
dense_input_size = len(dense_cols)
sparse_input_size = len(sparse_cols)
time_series_input_size = len(time_cols)

# 一个struct, 用于记录最后一天的数据
# 用于预测那15天的数据
class Last_Day_Sample():
    def __init__(self, sparse, dense, time_series):
        self.sparse = sparse
        self.dense = dense
        self.time_series = time_series

# 单只股票的数据集
class Single_Stock_Dataset(Dataset):
    def __init__(self, 
                 dense_X,   # 稠密数据标签
                 sparse_X,  # 离散数据标签(暂时没有用到)
                 time_X,    # 时间序列标签
                 Y,         # 标签(目前是收盘价和开盘价)
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
        return self.last_day.dense.unsqueeze(0)
    
    def last_sparse(self, device):
        return self.last_day.sparse.unsqueeze(0)
    
    def last_time_series(self, device):
        return self.last_day.time_series.unsqueeze(0)
    

# 用于构建多只股票的数据集
# 由于没有处理股票分类数据等分类标签, 所以暂时没有投入使用
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
    symbol: 股票编号, device: 使用的训练设备
    功能: 为symbol对应的股票创建数据集, 使用day, week, month字段
    
    Todo List:
    120m 60m等周期的数据使用
    '''

    # 读取csv并转化为ndarray
    sparse_X = np.array([],dtype=np.int64).reshape(0,len(sparse_cols))
    dense_X = np.array([],dtype=np.float64).reshape(0,len(dense_cols))
    time_X = np.array([],dtype=np.float64).reshape(0,len(time_cols))


    # 综合收盘价和开盘价, Y的前15个元素是收盘价, 后15个是开盘价
    label_Y = np.array([],dtype=np.float64).reshape(0, 30)

    for period in ["day","week","month"]:
        csv_name = symbol + "_" + period + ".csv"
        csv_name = os.path.join(".\\data\\stocks",symbol,csv_name)
        
        df = pd.read_csv(csv_name,encoding="utf-8")
        # 缺失值处理:填充为0
        # 若使用ffill可能会再次填充NaN
        df = df.fillna(0)

        #读取离散数据
        sparse_sample = np.array(df[sparse_cols],dtype=np.int64)

        #读取连续数据
        dense_sample = np.array(df[dense_cols],dtype=np.float64)

        # 读取时间序列数据
        time_sample = np.array(df[time_cols],dtype=np.float64)

        #读取收盘价
        close_price = np.array(df["close"],np.float64)
        open_price = np.array(df["open"],np.float64)
        if(period == "day"):
            # 将15天的数据合为一个Y label
            final_index = len(df) - 15
            label_sample = np.array([close_price[i:i+15] for i in range(final_index)], dtype=np.float64)
            label_sample = np.hstack((label_sample, np.array([open_price[i:i+15] for i in range(final_index)], dtype=np.float64)))

            last_day = Last_Day_Sample(sparse_sample[-1],dense_sample[-1],time_sample[-1])


        elif(period == "week"):
            # 将2周的数据合为一个Y lebel
            final_index = len(df) - 1
            label_sample = np.array([],dtype=np.float64).reshape(0, 30)
            for i in range(final_index):
                cf7 = np.full(7,close_price[i], dtype=np.float64)
                cb7 = np.full(7,close_price[i+1], dtype=np.float64)
                cavg = (close_price[i]+close_price[i+1])/2

                of7 = np.full(7,open_price[i], dtype=np.float64)
                ob7 = np.full(7,open_price[i+1], dtype=np.float64)
                oavg = (open_price[i]+open_price[i+1])/2

                # 此处一个Y的结构为[前一周, 两周均值, 后一周]
                label_sample = np.vstack((label_sample, np.hstack((cf7,cavg,cb7,of7,oavg,ob7), dtype=np.float64)))

        elif(period == "month"):
            c = [np.full(15,value) for value in close_price]
            o = [np.full(15,value) for value in open_price]

            # 这里后续15天的数据认定为第1天的数据
            label_sample = np.hstack((c,o),dtype=np.float64)

        # 将读取到的每个样本拼接到总训练矩阵后面
        sparse_X = np.vstack((sparse_X, sparse_sample[:len(label_sample)]))
        dense_X = np.vstack((dense_X, dense_sample[:len(label_sample)]))
        time_X = np.vstack((time_X, time_sample[:len(label_sample)]))
        label_Y = np.vstack((label_Y, label_sample))

    # 把最后一天的数据补充在最后, 因为要同时进行归一化
    # 构建训练集时不使用最后一天的数据
    sparse_X = np.vstack((sparse_X, last_day.sparse))
    dense_X = np.vstack((dense_X, last_day.dense))
    time_X = np.vstack((time_X, last_day.time_series))


    # 数据归一化
    dense_X = normalize(dense_X,"max",axis=0)
    time_X = normalize(time_X,"max",axis=0)


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