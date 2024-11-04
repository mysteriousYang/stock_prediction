# -*- coding:utf-8 -*-
import requests
import os
from file_utility import check_paths,exist_path,storge_stock
from logger import Enable_Logger
from conf import headers,proxies


def get_stocks():
    '''
    该函数用于获取股票的基本信息, 返回得到的股票代号列表
    注: 由于爬取雪球的接口需要cookies, 可能导致代码无法运行
        需要更新cookie, 即可运行.
        同时, 因为测试机上有代理,所以指定了proxies字段
        如有必要可将其删除.
    '''
    # 可以获得多页股票的数据，不过本题只需50支股票，页数为1即可
    for pge in range(1):
        url = 'https://xueqiu.com/service/v5/stock/screener/quote/list?page=%d&size=50&order=desc&orderby=percent&order_by=percent&market=CN&type=sh_sz&_=1623304455997' % (pge)
        response = requests.get(url=url, headers=headers, proxies=proxies)
        json_data = response.json()

    exist_path(".\\data\\stocks")
    stocks_list = list()

    for stock in json_data["data"]["list"]:
        storge_stock(stock)
        stocks_list.append(stock["symbol"])

    # 获得三大指数
    # SH000001 上证指数
    # SZ399001 深证指数
    # SZ399006 创业板指
    url = "https://stock.xueqiu.com/v5/stock/batch/quote.json?symbol=SH000001,SZ399001,SZ399006"
    response = requests.get(url=url, headers=headers, proxies=proxies)
    json_data = response.json()
    for i in range(3):
        storge_stock(json_data["data"]["items"][i]["quote"])
        stocks_list.append(json_data["data"]["items"][i]["quote"]["symbol"])

    return stocks_list
        

if __name__ == "__main__":
    check_paths()
    Enable_Logger()

    get_stocks()