# -*- coding:utf-8 -*-
# 这个文件存放模型相关的代码
# 例如模型定义 模型训练 模型预测等
import os
import sys
import torch
import sklearn
import json
import numpy as np
import torch.nn as nn
from logger import Enable_Logger
from file_utility import check_paths,exist_path
from dataset import build_single_dataset
from dataset import dense_input_size,sparse_input_size,time_series_input_size
from torch.utils.data import DataLoader

# 使用GPU进行训练
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 模型定义
class MultiStepLSTMWithEmbedding(nn.Module):
    '''
    这里借鉴了DLRM模型的结构, 对于分类数据, 采用embedding层将其转化为稠密数据
    稠密数据输入到普通的全连接神经网络.
    对于时间序列数据(其实本题的很多dense数据也可更改为时间序列数据), 我选择将其
    纳入LSTM网络, 当然也可以使用RNN与CNN网络增强当前模型的效果,
    RNN擅长处理时间序列任务, 而CNN可以从更高的维度获取数据的联系

    DLRM网络的架构简图
    output:
                            vector of values
    model:                        |
                                 /\
                                /__\
                                |
        _____________________> Op  <___________________
     /                         |                      \
     /\                        /\                      /\
    /__\                      /__\           ...      /__\
    |                          |                       |
    |                         Op                      Op
    |                    ____/__\_____           ____/__\____
    |                   |_Emb_|____|__|    ...  |_Emb_|__|___|
    input:
    [ dense features ]     [sparse indices] , ..., [sparse indices]

    对于后续15天数据的预测, 我选择直接在建立数据集的时候将后续15天的数据作为label
    (详情参见dataset的构建代码)
    也可以使用迭代预测的办法, 先预测1天的数据, 再将这一天的数据迭代预测后续若干
    天的label

    Todo List:
    使用普通机器学习模型(如RF, SVM)对比预测
    使用生成式模型(如GAN)生成预测数据
    部分数据loss过大
    用于单独处理三个指数的模型
    '''
    def __init__(self, 
                 dense_input_size, 
                 sparse_input_size, 
                 sparse_embedding_dims,
                 time_series_input_size, 
                 hidden_size=50, 
                 num_layers=3, 
                 output_steps=10):
        super(MultiStepLSTMWithEmbedding, self).__init__()
        # 注:这里数据使用的float64,所以网络层的数据类型都需要修改成float64
        
        # 稀疏特征的嵌入层，输入为类别数，输出为嵌入维度
        self.embeddings = nn.ModuleList([nn.Embedding(num_categories, emb_dim) 
                                         for num_categories, emb_dim in sparse_embedding_dims])
        
        # 稠密特征的全连接层
        dense_emb_size = dense_input_size + sum([emb_dim for _, emb_dim in sparse_embedding_dims])
        # 由于稠密特征不一定全部有效(例如三个指数), 所以增加了ReLU层
        self.fc_dense1 = nn.Linear(dense_emb_size, 30, dtype=torch.float64)
        self.fc_dense2 = nn.Linear(30, 25, dtype=torch.float64)
        self.relu_dense = nn.ReLU()
        self.fc_dense3 = nn.Linear(25, 20, dtype=torch.float64)

        
        # 时间序列特征的LSTM层
        self.lstm = nn.LSTM(time_series_input_size, hidden_size, num_layers, batch_first=True, dtype=torch.float64)
        self.fc_lstm = nn.Linear(hidden_size, 10, dtype=torch.float64)
        
        # 最终输出层，合并所有特征并输出未来5天的价格
        self.fc_out1 = nn.Linear(30, 20, dtype=torch.float64)
        self.fc_out2 = nn.Linear(20, output_steps, dtype=torch.float64)

    def forward(self, dense_inputs, sparse_inputs, time_series):
        # 嵌入层处理稀疏特征
        # sparse_embedded = [emb_layer(sparse_inputs[:, i]) for i, emb_layer in enumerate(self.embeddings)]
        # sparse_embedded = torch.cat(sparse_embedded, dim=1)
        
        # 稠密特征与嵌入特征合并
        # dense_combined = torch.cat([dense_inputs, sparse_embedded], dim=1)
        # dense_out = self.fc_dense(dense_combined)
        
        # 此处没有用到sparse所以不启用dense_combined
        dense_out = self.fc_dense1(dense_inputs)
        dense_out = self.relu_dense(dense_out)
        dense_out = self.fc_dense2(dense_out)
        dense_out = self.fc_dense3(dense_out)
        
        # LSTM处理时间序列特征
        lstm_out, _ = self.lstm(time_series)
        # lstm_out = self.fc_lstm(lstm_out[:, -1, :])  # 获取最后一个时间步的输出
        lstm_out = self.fc_lstm(lstm_out)
        
        # 合并所有特征并输出15天预测
        combined = torch.cat((dense_out, lstm_out), dim=1)
        combined = self.fc_out1(combined)
        return self.fc_out2(combined)



def train(dataset,num_epochs=30,lr=0.01):
    '''
    该函数用于训练模型并存储
    默认epochs为30
    '''

    # 以后可能用上的有关分类数据的设置
    sparse_input_size = 0
    sparse_embedding_dims = []
    # sparse_embedding_dims = [(10, 5), (15, 4)]  # 2个稀疏特征的类别数量和嵌入维度
   

    # 小批量大小设置
    batch_size = 16

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 初始化模型和优化器
    model = MultiStepLSTMWithEmbedding(dense_input_size=dense_input_size,
                                    sparse_input_size=sparse_input_size,
                                    sparse_embedding_dims=sparse_embedding_dims,
                                    time_series_input_size=time_series_input_size).to(device)
    # 使用MSE评价函数与Adam优化方法
    # 此处增加了0.1的权重衰减, 用于丢弃某些不重要的神经元
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.1)

    # 训练模型
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for dense_batch, sparse_batch, time_series_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(dense_batch, sparse_batch, time_series_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if(epoch % 10 == 0):
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}")

    # 存储已经训练好的模型
    print("训练完成")
    exist_path(".\\data\\model")
    torch.save(
        model,
        f".\\data\\model\\{dataset.symbol}.pth"
    )
    print(f"\\model\\{dataset.symbol}.pth 已保存")
    
    return model


def predict(symbol:str, epochs=30, lr=0.001):
    '''
    这个函数用于预测代码为symbol的股票
    将预测好的结果存为json
    如果没有训练则训练一遍后存储
    如果有训练好的模型则直接读取训练好的模型
    '''
    train_dataset = build_single_dataset(symbol, device)
    stock_name = train_dataset.name

    exist_path(".\\data\\model")
    exist_path(".\\data\\prediction\\")
    model_path = f".\\data\\model\\{symbol}.pth"

    # 查看是否存在已经训练好的模型
    if(os.path.exists(model_path)):
        model = torch.load(model_path)
    else:
        model = train(train_dataset,epochs,lr)

    # 将模型切换为预测模式并进行预测
    model.eval()
    with torch.no_grad():
        result = model(train_dataset.last_dense(device),
                    train_dataset.last_sparse(device),
                    train_dataset.last_time_series(device)).cpu().numpy()
        result = result.flatten()
        print(f"{symbol} {stock_name} 股票的接下来5天收盘价预测为")
        print(result[:5])
        print(f"{symbol} {stock_name} 股票的接下来5天开盘价预测为")
        print(result[5:])

    # 存储预测结果为json
    dump_path = f".\\data\\prediction\\{symbol}"
    exist_path(dump_path)

    json_data = dict()
    json_data["name"] = stock_name
    json_data["symbol"] = symbol
    json_data["close"] = list(result[:5])
    json_data["open"] = list(result[5:])

    with open(os.path.join(dump_path,f"{symbol}.json"),"w",encoding="utf-8") as fp:
        json.dump(json_data, fp, ensure_ascii=False)

    return json_data


if __name__ == "__main__":
    check_paths()
    Enable_Logger()

    predict("SH600221")
    pass