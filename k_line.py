# -*- coding:utf-8 -*-
import requests
import os
from file_utility import exist_path,storge_stock_k_line
from xueqiu_api import k_line_url
from conf import headers,proxies


def get_k_line_data():
    exist_path(".\\data\\k_line")
    period_list = ["1d","day","week","month","year"]

    for symbol in os.listdir(".\\data\\stocks"):
        #do sth

        for period in period_list:
            url = k_line_url(symbol, period)
            response = requests.get(url,proxies=proxies,headers=headers)
            json_data = response.json()

            storge_stock_k_line(symbol,json_data["data"],period)
    pass

if __name__ == "__main__":
    get_k_line_data()