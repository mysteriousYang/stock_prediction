# -*- coding:utf-8 -*-
import os
import sys
import datetime

'''
这是一个输出记录器
用于将每次运行的控制台结果记录到磁盘
默认日志目录为./logs
默认名称为运行时间
'''

LOG_FILE = ".\\logs\\" + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + ".log"

class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w')
 
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
 
    def flush(self):
        pass

def Enable_Logger():
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE,"w") as fp:
            pass
    sys.stdout = Logger(LOG_FILE, sys.stdout)