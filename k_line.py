# -*- coding:utf-8 -*-
# 这个文件用于获取股票的详细信息
# 主要是获取一些k线数据
import requests
import os
from file_utility import check_paths,storge_stock_k_line,stock_to_csv
from xueqiu_api import k_line_url
from logger import Enable_Logger
from conf import headers,proxies


def get_k_line_data():
    '''
    该函数用于从雪球下载每只股票的详细信息
    会自动下载/data/stocks文件夹下所有股票的详细信息
    将每个不同的period存储为不同的json
    注: 由于爬取雪球的接口需要cookies, 可能导致代码无法运行
        需要更新cookie, 即可运行.
        同时, 因为测试机上有代理,所以指定了proxies字段
        如有必要可将其删除.

    Todo List:
    多线程下载器
    '''

    period_list = ["1d","day","week","month","year"]

    for symbol in os.listdir(".\\data\\stocks"):
        #do sth

        for period in period_list:
            url = k_line_url(symbol, period)
            response = requests.get(url,proxies=proxies,headers=headers)
            json_data = response.json()

            storge_stock_k_line(symbol,json_data["data"],period)
    pass

def transfer_csv():
    '''
    这个函数用于将获取到的股票k线数据json文件转化为便于构建数据集的csv
    '''
    for symbol in os.listdir(".\\data\\stocks"):

        for file_name in os.listdir(f".\\data\\stocks\\{symbol}"):
            if("info" in file_name):
                continue
            u_pos = file_name.find('_') + 1
            d_pos = file_name.find('.')
            period = file_name[u_pos:d_pos]
            stock_to_csv(symbol, period)
    pass

if __name__ == "__main__":
    check_paths()
    Enable_Logger()

    # get_k_line_data()
    transfer_csv()
    pass