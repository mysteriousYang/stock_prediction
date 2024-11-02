# -*- coding:utf-8 -*-
import json
import os

_minute_columns = [
    "current",
    "volume",
    "avg_price",
    "chg",
    "percent",
    "timestamp",
    "amount",
    "high",
    "low",
    "amount_total",
    "volume_total",
    "macd",
    "kdj",
    "ratio",
    
    "capital_small",
    "capital_medium",
    "capital_large",
    "capital_xlarge",
    
    "volume_sum",
    "volume_sum_last"
]

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


def stock_to_csv(stock_symbol:str, period:str):
    '''
    period可选: 1d, 5d, day, week, month, quarter, year
               120m, 60m, 30m, 15m, 5m, 1m
    '''
    load_path = os.path.join(".\\data\\stocks", stock_symbol)
    json_file = stock_symbol + "_" + period + ".json"

    if(period in ("1d","5d")):
        with open(os.path.join(load_path, json_file), "r", encoding="utf-8") as fp:
            json_data = json.load(fp)

        csv_file = stock_symbol + "_" + period + ".csv"
        with open(os.path.join(load_path, csv_file), "w", encoding="utf-8") as fout:
            fout.write(",".join(_minute_columns) + "\n")

            for sample in json_data["items"]:
                data = list()
                for key in _minute_columns:
                    if(key in ("capital_small","capital_medium","capital_large","capital_xlarge")):
                        #有些可能没有capital字段
                        if(sample["capital"] is None):
                            obj = None
                        else:
                            obj = sample["capital"][key[8:]]
                    elif(key in ("volume_sum","volume_sum_last")):
                        #有些可能没有volume compare字段
                        if(sample["volume_compare"] is None):
                            obj = None
                        else:
                            obj = sample["volume_compare"][key]
                    else:
                        obj = sample[key]
                    

                    data.append(str(obj))
                fout.write(",".join(data) + "\n")
        print(csv_file, "已转换")
        return
    
    else:
        with open(os.path.join(load_path, json_file), "r", encoding="utf-8") as fp:
            json_data = json.load(fp)

        csv_file = stock_symbol + "_" + period + ".csv"
        with open(os.path.join(load_path, csv_file), "w", encoding="utf-8") as fout:

            column = json_data["column"]
            fout.write(",".join(column) + "\n")

            for sample in json_data["item"]:
                fout.write(",".join([str(elem) for elem in sample]) + "\n")
        print(csv_file, "已转换")
        return