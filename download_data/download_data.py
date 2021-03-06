#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Time : 2021/6/11 15:26
# Author : 
# File : utils.py
# Software: PyCharm
# description:
import os
import sys
import datetime


def execCmd(cmd):
    r = os.popen(cmd)
    text = r.read()
    r.close()
    return text.strip('\n')


# 拉取一个小时的数据，进行去重
def download_data(date, hour):
    tmpPostfix = "_tmp"
    execCmd("sh /data/recom/recall/tfgpu/download_data.sh " + date + " " + hour + " " + tmpPostfix)


if __name__ == "__main__":
    date = sys.argv[1]
    hour = sys.argv[2]
    t = datetime.datetime.now()
    download_data(date, hour)
    print('cost time:', datetime.datetime.now() - t)
