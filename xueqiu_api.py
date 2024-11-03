# -*- coding:utf-8 -*-
#该文件用于记录雪球的股票api
import time
from logger import Enable_Logger

XUEQIU_COOKIE = "cookiesu=381730628456277; device_id=80a0eabf4f91101e7f916fd161c2c59d; remember=1; xq_a_token=5744d5dda6b2a210d801920ff81e1d51b07cbe72; xqat=5744d5dda6b2a210d801920ff81e1d51b07cbe72; xq_id_token=eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJ1aWQiOjY1OTIxMTA4MjUsImlzcyI6InVjIiwiZXhwIjoxNzMyOTQ3MzMxLCJjdG0iOjE3MzA2Mjg0OTM1ODAsImNpZCI6ImQ5ZDBuNEFadXAifQ.pQZlJbUxFa94eVHghNu5_OYZHBGPTBiJWvdG3lI_HbS4nN4GQ-h1jTwojcDKYEy7aEw_JnYNJxLdA_OF3Fr-k3O5ItZ2Ei0UP_tWV-8XBT2tvAlVMt4VfkFlEicTKcJTz7enD8u68LjnVHmGjWlUwHNJQAuPSbkFP_7j0KpTQ67g3HeoaWkR9AQAXpSCvUoSwyNHK6cMlYb1mctOutLxYcg8slcr2fZtOXxge45gcGihfbuJ0xS0nOWpEtnzgIiGIT4B5JlMVxxBQNwTvm3_5IfvXwu43SDaeofILqOk8kVTpHwPDS7ISiRqqbYOzCXd3ICak-wOwFijacW_sckT1g; xq_r_token=5ca2c30be941cf82d4cd89ea71b227fb254c05e3; xq_is_login=1; u=6592110825; is_overseas=0; ssxmod_itna=YqIx2Dc7DtK=dGHqGdD7AwAxCq247qe+PD=epbYkDlpexA5D8D6DQeGTrnEFTFqYhDxdbe9GGDYFdDaRaRpCjiGpPLk+DB3DEx065V+DYYCDt4DTD34DYDixibzxi5GRD0KDF7dy/1yDYPDE05DR2PDuPYhDGa1jOFhDeKD0oqHDQK+XOxDBOxAIDR7GiYDe2ahchgCe7fxK0KD9hYDshi6F0Kp9+LPMOfL03YdIx0kKq0OyZAC7b2kZhvGTlnDpehNiexrt0DDgW4at=24YW4YGl6g/mGPQ7h55maMOqDW7yUDD; ssxmod_itna2=YqIx2Dc7DtK=dGHqGdD7AwAxCq247qe+PD=epbYD668D440vB403qDg0aV3wHqAPey3OoKiNx2rAKxOD=WQBdk5Fx7QwDFqG7FeD"

def time_stamp():
    return int(time.time()*1000)

# def k_line_minute_url(symbol:str, period:str="1d"):
#     '''
#     period可选: 1d, 5d
#     '''
#     return f"https://stock.xueqiu.com/v5/stock/chart/minute.json?symbol={symbol}&period={period}"

def k_line_url(symbol:str, period:str="day"):
    '''
    period可选: 1d, 5d, day, week, month, quarter, year
               120m, 60m, 30m, 15m, 5m, 1m
    '''
    if(period in ("1d", "5d")):
        return f"https://stock.xueqiu.com/v5/stock/chart/minute.json?symbol={symbol}&period={period}"
    else:
        return f"https://stock.xueqiu.com/v5/stock/chart/kline.json?symbol={symbol}&begin={time_stamp()}&period={period}&type=before&count=-284&indicator=kline,pe,pb,ps,pcf,market_capital,agt,ggt,balance"


if __name__ == "__main__":
    Enable_Logger()

    print(time_stamp())
    pass