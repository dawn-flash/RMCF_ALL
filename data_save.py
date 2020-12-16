#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2020/9/24 15:16
# @Author  : Lelsey
# @File    : data_save.py
#数据保存模块，保存所有的实验数据
import os
import time
import json
import sys


class DataSave():
    def __init__(self,dir_name):
        """
        :param dir_name: 目录名，实际为函数名
        """
        self._dir_path=self.dir_vertify(dir_name)
        data_time=time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) #获取格式化时间
        self._file_path=os.path.join(self._dir_path,data_time+'.json')  #实际的文件完整路径
        pass

    def dir_vertify(self,dir_name):
        """
        判断目录是否存在，目录不存在则创建目录
        :param dir_name: 目录名
        :return: 文件路径
        """
        dir_path=os.path.join(os.getcwd(),'exper_result',dir_name)
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)
        return dir_path

    def sava_all_data(self,exper_data):
        """
        存储文件
        :param exper_data: 需要存储的数据
        :return:
        """
        print("文件开始保存{0}".format(self._file_path))
        with open(self._file_path,'w')as fw:
            json.dump(exper_data,fw,indent=4)
        print("文件保存结束！")


def test_data():
    data={"name":"",
          "age":18,
          "grade":[100,100],
          '(1,2)':1}
    function_name=sys._getframe().f_code.co_name
    print("函数名",function_name)

    ds=DataSave(function_name)
    ds.sava_all_data(data)

if __name__ == '__main__':
    test_data()