# -*- coding:utf-8 -*-
import json
import os

def exist_path(path:str):
    if(os.path.exists(path)):
        pass
    else:
        os.mkdir(path)
    return

def check_paths():
    exist_path(".\\data")
    exist_path(".\\logs")

def storge_stock(stock_obj):
    save_dir = os.path.join(".\\data\\stocks", stock_obj["symbol"])
    file_name = stock_obj["symbol"] + "_info.json"
    exist_path(save_dir)

    with open(os.path.join(save_dir, file_name),"w",encoding="utf-8") as fp:
        json.dump(stock_obj, fp, ensure_ascii=False)

    print(file_name, "已保存")

def storge_stock_k_line(stock_symbol:str, stock_obj, period:str):
    save_path = os.path.join(".\\data\\stocks", stock_symbol)
    exist_path(save_path)

    file_name = stock_symbol + "_" + period + ".json"
    with open(os.path.join(save_path, file_name), "w", encoding="utf-8") as fp:
        json.dump(stock_obj, fp, ensure_ascii=False)
    print(file_name, "已保存")