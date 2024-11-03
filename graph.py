# -*- coding:utf-8 -*-
# 这个文件用于存放有关图表的代码
# 目前主要为预测结果的图表
import os
import matplotlib.pyplot as plt
import matplotlib
import json
from shutil import copyfile
from file_utility import exist_path
from models import predict

def draw_and_save(predict_json):
    '''
    该函数接受predict的json结果, 并将15天的收盘价和开盘价存储为png
    '''
    save_path = f".\\data\\prediction\\{predict_json['symbol']}"
    exist_path(save_path)

    # 收盘价绘图
    graph_path = os.path.join(save_path, f"{predict_json['symbol']}_close.png")

    x = range(1,16)
    y = predict_json["close"]
    matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    fontdict={'fontname': 'Microsoft YaHei', 'fontsize': 12}

    plt.plot(x, y, marker='o', color='b', linestyle='-', linewidth=2, markersize=5)
    plt.xlabel("天数",fontdict=fontdict)
    plt.ylabel("收盘价",fontdict=fontdict)
    plt.title(
        f"{predict_json['symbol']} {predict_json['name']} 15天收盘价预测", fontdict=fontdict)
    for i in range(len(x)):
        plt.text(x[i], y[i], f'{y[i]:.2f}', ha='center', va='bottom', fontsize=8, color='black')

    plt.savefig(graph_path, dpi=300, bbox_inches='tight')
    print(graph_path," 已保存")
    plt.close()


    # 开盘价绘图
    graph_path = os.path.join(save_path, f"{predict_json['symbol']}_open.png")

    x = range(1,16)
    y = predict_json["open"]
    matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    fontdict={'fontname': 'Microsoft YaHei', 'fontsize': 12}

    plt.plot(x, y, marker='o', color='b', linestyle='-', linewidth=2, markersize=5)
    plt.xlabel("天数",fontdict=fontdict)
    plt.ylabel("开盘价",fontdict=fontdict)
    plt.title(
        f"{predict_json['symbol']} {predict_json['name']} 15天开盘价预测", fontdict=fontdict)
    for i in range(len(x)):
        plt.text(x[i], y[i], f'{y[i]:.2f}', ha='center', va='bottom', fontsize=8, color='black')

    plt.savefig(graph_path, dpi=300, bbox_inches='tight')
    print(graph_path," 已保存")
    plt.close()

def merge_graphs():
    '''
    该函数用于收集 /data/prediction/* 中的图片
    将其存放到某个固定目录
    免得查看一张图片还得挨个点进去
    '''
    exist_path(".\\data\\graph_predicted")

    for root, dirs, files in os.walk(".\\data\\prediction"):
        for file in files:
            if(file[-4:] == ".png"):
                copyfile(
                    os.path.join(root,file),
                    f".\\data\\graph_predicted\\{file}"
                )
                print(f"已传送 {file}")


if __name__ == "__main__":
    merge_graphs()

    pass