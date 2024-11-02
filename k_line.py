# -*- coding:utf-8 -*-
import requests
import os
from file_utility import check_paths,storge_stock_k_line,stock_to_csv
from xueqiu_api import k_line_url
from logger import Enable_Logger
from conf import headers,proxies


def get_k_line_data():
    '''
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