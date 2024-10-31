# -*- coding:utf-8 -*-
import os
import sys
import datetime

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