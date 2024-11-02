# -*- coding:utf-8 -*-
import numpy as np

if __name__ == "__main__":
    a = np.array([]).reshape(0,3)
    a = np.vstack((a,[1,2,3]))
    a = np.vstack((a,[6,7,8]))
    print(a)
    pass