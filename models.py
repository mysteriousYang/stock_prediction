# -*- coding:utf-8 -*-
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
from torch.utils.data import DataLoader,Dataset,TensorDataset

# 使用GPU进行训练
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 模型定义
class MultiStepLSTMWithEmbedding(nn.Module):
    def __init__(self, 
                 dense_input_size, 
                 sparse_input_size, 
                 sparse_embedding_dims,
                 time_series_input_size, 
                 hidden_size=50, 
                 num_layers=2, 
                 output_steps=15):
        super(MultiStepLSTMWithEmbedding, self).__init__()
        # 注:这里数据使用的float64,所以网络层的数据类型都需要修改成float64
        
        # 稀疏特征的嵌入层，输入为类别数，输出为嵌入维度
        self.embeddings = nn.ModuleList([nn.Embedding(num_categories, emb_dim) 
                                         for num_categories, emb_dim in sparse_embedding_dims])
        
        # 稠密特征的全连接层
        dense_emb_size = dense_input_size + sum([emb_dim for _, emb_dim in sparse_embedding_dims])
        self.fc_dense = nn.Linear(dense_emb_size, 20, dtype=torch.float64)

        
        # 时间序列特征的LSTM层
        self.lstm = nn.LSTM(time_series_input_size, hidden_size, num_layers, batch_first=True, dtype=torch.float64)
        self.fc_lstm = nn.Linear(hidden_size, 10, dtype=torch.float64)
        
        # 最终输出层，合并所有特征并输出未来15天的价格
        self.fc_out = nn.Linear(30, output_steps, dtype=torch.float64)

    def forward(self, dense_inputs, sparse_inputs, time_series):
        # 嵌入层处理稀疏特征
        # sparse_embedded = [emb_layer(sparse_inputs[:, i]) for i, emb_layer in enumerate(self.embeddings)]
        # sparse_embedded = torch.cat(sparse_embedded, dim=1)
        
        # 稠密特征与嵌入特征合并
        # dense_combined = torch.cat([dense_inputs, sparse_embedded], dim=1)
        # dense_out = self.fc_dense(dense_combined)
        
        # 此处没有用到sparse所以不启用dense_combined
        # print(dense_inputs)
        dense_out = self.fc_dense(dense_inputs)
        
        # LSTM处理时间序列特征
        lstm_out, _ = self.lstm(time_series)
        # print(lstm_out.shape)
        # lstm_out = self.fc_lstm(lstm_out[:, -1, :])  # 获取最后一个时间步的输出
        lstm_out = self.fc_lstm(lstm_out)
        # print(lstm_out.shape)
        
        # 合并所有特征并输出15天预测
        combined = torch.cat((dense_out, lstm_out), dim=1)
        return self.fc_out(combined)



def train(dataset):
    # 示例数据准备
    # 假设有3个稠密特征、2个稀疏特征（每个特征的类别数量为10和15，嵌入维度分别为5和4），以及5天窗口的时间序列特征
    # dense_input_size = 3
    sparse_input_size = 0
    # sparse_embedding_dims = [(10, 5), (15, 4)]  # 2个稀疏特征的类别数量和嵌入维度
    # time_series_input_size = 2  # 每天的时间序列特征数量（如收盘价和成交量）
    sparse_embedding_dims = []

    # # 假设数据集大小为1000
    batch_size = 16
    # X_dense = torch.randn(1000, dense_input_size).to(device)  # 稠密特征输入
    # X_sparse = torch.randint(0, 10, (1000, sparse_input_size)).to(device)  # 稀疏特征输入
    # X_time_series = torch.randn(1000, 5, time_series_input_size)  # 时间序列特征输入
    # y = torch.randn(1000, 15)  # 标签（未来15天的收盘价）

    # 构建数据加载器
    # train_dataset = TensorDataset(X_dense, X_sparse, X_time_series, y)

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 初始化模型和优化器
    model = MultiStepLSTMWithEmbedding(dense_input_size=dense_input_size,
                                    sparse_input_size=sparse_input_size,
                                    sparse_embedding_dims=sparse_embedding_dims,
                                    time_series_input_size=time_series_input_size).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    model.train()
    num_epochs = 30
    for epoch in range(num_epochs):
        total_loss = 0
        for dense_batch, sparse_batch, time_series_batch, y_batch in train_loader:
            optimizer.zero_grad()
            # print(dense_batch.shape,sparse_batch.shape,time_series_batch.shape)
            y_pred = model(dense_batch, sparse_batch, time_series_batch)
            # print(y_pred, y_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if(epoch % 10 == 0):
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}")

    print("训练完成")
    exist_path(".\\data\\model")
    torch.save(
        model,
        f".\\data\\model\\{dataset.symbol}.pth"
    )
    print(f"\\model\\{dataset.symbol}.pth 已保存")
    
    return model


def predict(symbol:str):
    train_dataset = build_single_dataset(symbol, device)
    stock_name = train_dataset.name

    exist_path(".\\data\\model")
    exist_path(".\\data\\prediction\\")
    model_path = f".\\data\\model\\{symbol}.pth"

    if(os.path.exists(model_path)):
        model = torch.load(model_path)
    else:
        model = train(train_dataset)

    model.eval()
    with torch.no_grad():
        result = model(train_dataset.last_dense(device),
                    train_dataset.last_sparse(device),
                    train_dataset.last_time_series(device)).cpu().numpy()
        result = result.flatten()
        print(f"{symbol} {stock_name} 股票的接下来15天收盘价预测为")
        print(result)

    # 存储预测结果为json
    dump_path = f".\\data\\prediction\\{symbol}"
    exist_path(dump_path)

    json_data = dict()
    json_data["name"] = stock_name
    json_data["symbol"] = symbol
    json_data["close"] = list(result)

    with open(os.path.join(dump_path,f"{symbol}.json"),"w",encoding="utf-8") as fp:
        json.dump(json_data, fp, ensure_ascii=False)

    return json_data


if __name__ == "__main__":
    check_paths()
    Enable_Logger()

    predict("SH600221")
    pass