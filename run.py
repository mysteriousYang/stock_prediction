# -*- coding:utf-8 -*-
import os
import json
import pandas
from stocks_get import get_stocks
from file_utility import check_paths
from logger import Enable_Logger
from k_line import get_k_line_data,transfer_csv
from models import predict
from graph import draw_and_save, merge_graphs

if __name__ == "__main__":
    check_paths()
    Enable_Logger()

    print("股市预测任务开始")

    print("正在获取股票信息")
    stocks_list = get_stocks()
    print("获取成功")

    print("正在获取详细信息")
    get_k_line_data()
    print("正在转换数据")
    transfer_csv()
    print("完成")

    # 如果已经下载好了股票信息, 可以把以上用于下载的代码注释掉
    # 直接从磁盘读取需要处理的股票代码列表
    # stocks_list = os.listdir(".\\data\\stocks")

    table = dict()
    table["close"] = dict()
    table["open"] = dict()

    print("正在开始训练和预测任务")
    for symbol in stocks_list:
        result = predict(symbol, 50)
        draw_and_save(result)
        table["close"][symbol] = result["close"]
        table["open"][symbol] = result["open"]

    merge_graphs()
    pandas.DataFrame(table["close"]).to_csv(".\\predict_close_result.csv", index=False)
    pandas.DataFrame(table["open"]).to_csv(".\\predict_open_result.csv", index=False)