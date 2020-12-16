#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2020/10/14 19:07
# @Author  : Lelsey
# @File    : test.py.py

import numpy as np

def test1():
    a=np.array([[1,2],[3,4]])
    b=np.array([1,2])
    print(a*b)
    print(a.dot(b))
    print(type(a.dot(b)))

if __name__ == '__main__':
    # a=np.array([[0,0,1,8.1,0.9,0.9,0.1],
    #             [0,0,1,8.1,0.9,0.9,0.1],
    #             [1,1,0,0,0,0,0],
    #             [8.1,8.1,0,0,0,0,0],
    #             [0.9,0.9,0,0,0,0,0],
    #             [0.9,0.9,0,0,0,0,0],
    #             [0.1,0.1,0,0,0,0,0]])
    # print("行列式的值：",np.linalg.det(a))
    # print("行列式的特征值",np.linalg.eigvals(a))
    test1()


