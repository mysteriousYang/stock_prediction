# -*- coding:utf-8 -*-
#该文件用于记录雪球的股票api
import time

XUEQIU_COOKIE = "acw_tc=2760827b17303555466103191eb667c7a9a9a22a1c08f44625d274ef5a16ea; cookiesu=531730355546619; device_id=80a0eabf4f91101e7f916fd161c2c59d; smidV2=20241031141910763baaad4bfabd6db8ee19edc83973280040f032e6af05d20; remember=1; xq_a_token=5744d5dda6b2a210d801920ff81e1d51b07cbe72; xqat=5744d5dda6b2a210d801920ff81e1d51b07cbe72; xq_id_token=eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJ1aWQiOjY1OTIxMTA4MjUsImlzcyI6InVjIiwiZXhwIjoxNzMyOTQ3MzMxLCJjdG0iOjE3MzAzNTU1OTY2MzUsImNpZCI6ImQ5ZDBuNEFadXAifQ.DAiB0E_uysAHuGFCY_DQ9R91rBEc3711dDbRYW3-lqRvytb0YTCz5HmjZo1uyvnBPCcjCoHDp1bTBib2piFgCSn47f6Ri_CF3-Y_OnAVHGw9s24uBMDtzI4iFdhsSJMuzgWQKPp-Bz41aag5pCbk0LSlYr84uPHIkO3_K9cyQwxxR3q31OcxG3MUw-I6OMbupyk5pmOsjxzusoxPoDoOnMCtZbONWm-SZYxTn3B0ZkgPV-wVbTPR4oYD8uaaakfVgJYZOcGJrJ4bmY2BAagSpYzbbbaWht-u6PhB9siNO7X1_LxjPKiJGE5Xrs8vNFT_06wsEdGkflBIYhQdvtAqhA; xq_r_token=5ca2c30be941cf82d4cd89ea71b227fb254c05e3; xq_is_login=1; u=6592110825; is_overseas=0; .thumbcache_f24b8bbe5a5934237bbc0eda20c1b6e7=XFvkbay8gOEDgArzrivwSYJIp4kFIfTMxWGK3M71XzGXYi29Xbyia0wcsFYMPkLTRRzC3bVed+5Rd1RQCxjUCA%3D%3D; acw_sc__v2=6723218dcadd12bccb9fe8f0e1f74136ead662ca; ssxmod_itna=eqUO0K4RxIxh8DzxmxB4DISQF7H9uu2cPD=Q3rpkqDl=YxA5D8D6DQeGTrcpDBChRpoNUefarKCz0Bids53Z+ET4s81xdpYDU4i8DCw2IoxxeWtD5xGoDPxDeDADYo0DAqiOD7qDdXLvvz2xGWDmRsDYvHDQ5h54DFBZF8Y4i7DDyQzx07d3yPDG5omWRxI10hDimHY+yI92ifqQ0kD7y+DlcqBz2kHIMUHBHf2tkh6mKDXpQDvEvmc6OPRxBzPbhWTWbex=hNrl0dtBw4plrWWBuqpzG4f5eQHQDie0GqC0=/HDDWpQi/qiDD==; ssxmod_itna2=eqUO0K4RxIxh8DzxmxB4DISQF7H9uu2cPD=Q3rpTD62/0D0y2w403pHDB7q5qnRDkHjh3K25BIeLexkVb7kDwb5dFCRqGTdRA7csQtWmB3GAmIm30bpjgee0qTUKNBpmwHVTxrZ8qudhjTYyYnIk+wAoQe04=eQSDEYDEkGexu6eLnKSYGG6eS5c+n6GKjRmKpR6UOi0BRdPMmaHyQKem4TC18Dcn8DS/=Im3Wv7lfQP01Ypj3AzDUOMAk2oF3QhRcjsR36XDfH+AKr3OxG2UYKDFqD2UiD=; s=bh1dijzo7w"

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
    print(time_stamp())
    pass