# -*- coding:utf-8 -*-
from xueqiu_api import XUEQIU_COOKIE

proxies = {
    "http":"127.0.0.1:7890",
    "https":"127.0.0.1:7890"
}

headers = {
    #byd 爬这玩意要加cookies
    "Cookie": XUEQIU_COOKIE,
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.90 Safari/537.36"
}