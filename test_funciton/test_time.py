#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Time : 2021/5/31 16:14
# Author : 
# File : test_time.py
# Software: PyCharm
# description:
"""
https://www.runoob.com/python/python-date-time.html
"""
import datetime
import time


def getTomorrow(dateStr):
    date = datetime.datetime.strptime(dateStr, '%Y%m%d')

    delta = datetime.timedelta(days=1)
    tomorrow = date + delta
    return tomorrow.strftime('%Y%m%d')


def get_next_date_hour(date_str, hour_str):
    if hour_str == "23":
        return getTomorrow(date_str), '00'
    else:
        hour_next_str = str(int(hour_str) + 1)
        if len(hour_next_str) < 2:
            return date_str, '0' + hour_next_str
        else:
            return date_str, hour_next_str


# ================时间戳=====================
def get_timeStamp():
    t0 = "2021052100"
    t1 = "2021052101"
    # 转化为 <class 'time.struct_time'>
    # time.struct_time(tm_year=2021, tm_mon=5, tm_mday=21, tm_hour=0,
    # tm_min=0, tm_sec=0, tm_wday=4, tm_yday=141, tm_isdst=-1)
    tt0 = time.strptime(t0, '%Y%m%d%H')
    tt1 = time.strptime(t1, '%Y%m%d%H')
    st0 = int(time.mktime(tt0))  # 1621526400
    st1 = int(time.mktime(tt1))  # 1621530000  1621533600




