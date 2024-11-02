# -*- coding:utf-8 -*-
import os
import sys
import torch
import sklearn
import numpy as np
from logger import Enable_Logger
from file_utility import check_paths
from torch.utils.data import DataLoader,Dataset




if __name__ == "__main__":
    check_paths()
    Enable_Logger()
    pass