#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Time : 2021/5/30 15:48
# Author : 
# File : pyspark_logs.py
# Software: PyCharm
# description:
"""
    不能.collect()，否则会变为list

    recom.load_expose_click: uid, id, ctime
        sid暂时没啥用
        需要去除点击，但是dur_time时间短，read_rate小的样本？
    步骤：1， 排序
    2. 去重
    3. aggregate
"""

from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession


def init_spark(name):
    spark = SparkSession \
        .builder \
        .appName(name) \
        .enableHiveSupport() \
        .config("spark.driver.host", "localhost") \
        .getOrCreate()
    return spark


def addItemToList(lst, val):
    lst.append(val)
    return lst


def mergeLists(lst1, lst2):
    return lst1 + lst2


def test(sc1):
    blank_list = []
    rdd = sc1.parallelize([('a', 'ITEM_2', '2018'), ('b', 'USER_1', '2010'), ('a', 'ITEM_1', '2017'), ('a', 'ITEM_3', '2019'),('b', 'UER', '2000')])
    rdd1 = rdd.sortBy(lambda x: x[2]).map(lambda x: (x[0], x[1]))
    rdd2 = rdd1.aggregateByKey(blank_list, addItemToList, mergeLists)
    return rdd2


sc = init_spark('name')
sql_str = "select sid, uid, id, ctime from recom.load_expose_click where pdate=20210528 and phour=00 and uid is not null and ctime is not null limit 10"
rdd = sc.sql(sql_str).rdd
rdd1 = rdd.sortBy(lambda x: x.ctime)  # 按ctime排序
rdd2 = rdd1.map(lambda x: (x.uid, x.id))
"""
print(type(rdd1))
print(type(rdd1.collect()))  # list
"""
blank_list = []

rdd3 = rdd2.aggregateByKey(blank_list, addItemToList, mergeLists)


