# -*- coding:utf-8 -*-
import requests
import os
from xueqiu_api import k_line_minute_url

from file_utility import check_paths,exist_path,storge_stock
from logger import Enable_Logger
from conf import headers,proxies


def get_stocks():
    check_paths()
    Enable_Logger()

    # 可以获得多页股票的数据，不过本题只需50支股票，页数为1即可
    for pge in range(1):
        url = 'https://xueqiu.com/service/v5/stock/screener/quote/list?page=%d&size=50&order=desc&orderby=percent&order_by=percent&market=CN&type=sh_sz&_=1623304455997' % (pge)
        response = requests.get(url=url, headers=headers,proxies=proxies)
        json_data = response.json()

        # print(response.text)

    exist_path(".\\data\\stocks")

    for stock in json_data["data"]["list"]:
        storge_stock(stock)
        

if __name__ == "__main__":
    get_stocks()