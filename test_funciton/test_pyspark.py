#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Time : 2021/5/28 20:58
# Author : 
# File : test_pyspark.py
# Software: PyCharm
# description:
"""
0604:
    一周：click， 一亿
    item: 350万
    user: 530万
"""
from pyspark.sql import SparkSession

import pyspark
from pyspark import SparkContext
from pyspark import SparkConf

# 0604====================================================
def init_spark():
    spark = SparkSession \
        .builder \
        .appName('name') \
        .enableHiveSupport() \
        .getOrCreate()
    return spark


# sql_str = 'select uid, id, ctime from recom.load_expose_click where concat(pdate,phour) between "2021052101" and "2021052201" and uid is not null and id is not null and ctime is not null and sch not in ("relateDocOrig", "relateVideoOrig")'
sc = init_spark()
# rdd1 = sc.sql(sql_str).rdd
# rdd2 = rdd1.sortBy(lambda line:line.ctime)
# rdd_item = rdd1.map(lambda line:)



conf = SparkConf().setAppName("lg").setMaster('local[4]')
sc = SparkContext.getOrCreate(conf)
rdd = sc.parallelize([('B', 1), ('B', 2), ('A', 3), ('A', 4), ('A', 5)])
rdd_kv = rdd.groupByKey().mapValues(len)
rdd_filter = rdd_kv.filter(lambda x:x[1]>2)
rdd_filter.collect()


sc1 = SparkContext()
sc1.bo



# ====================================================
def add(a, b):
    c = a + b
    return c


def reduce_group(sc):
    rdd = sc.parallelize([('a', 3), ('b', 1), ('a', 1), ('a', 1), ('b', 1), ('b', 1)])
    # [('b', 3), ('a', 5)]
    print('reducebykey(add)', rdd.reduceByKey(add).collect())  # value. add, 必须 有func
    print('groupByKey()', rdd.groupByKey().mapValues(len).collect())  # 个数
    # [('b', 3), ('a', 3)]


def aggregate_func_def(num):  # https://www.cnblogs.com/LHWorldBlog/p/8215529.html
    """
    num不同的时候，分成的partiton不同，导致第一步不同
    num=1:  原始  seqFunc:max                 combine:+
        (1,1)       1,[1,2,7] -->  1,7        1,7   2:4
        (1,2)       2,[1,3,4]  --> 2,4
        (2, 1),
        (2, 3),
        (2, 4),
        (1, 7)
    num=2:   ori         max             combine +
            (1,1)    1,[1,2]-->1,3      1,[3,7] -->1,10   2,[3,4]-->2,7
            (1,2)    2,[1]  -->2,3
            (2, 1),
            --------
            (2, 3),  2,[3,4] -->2,4
            (2, 4),
            (1, 7)   1,[7]  --> 1,7

    num=3:   ori         max                combine +
            (1,1)    1,[1,2]-->1,3      1,[3,7] -->1,10   2,[3,4]-->2,7
            (1,2)
            --------
            (2, 1)   2,[1,3] -->2,3
            (2, 3)
            --------
            (2, 4)   2,[4] -->2,4
            (1, 7)   1,[7]  --> 1,7

    num=6:  原始      seqFunc:max     combine:+
            (1,1)       1,3             1: 3+3+7=13   2:3+3+4=10
            (1,2)       1,3
            (2, 1),     2,3
            (2, 3),     2,3
            (2, 4),     2,4
            (1, 7)      1,7

    :param num:
    :return:
    """

    def seqFunc(a, b):
        return max(a, b)  # 取最大值

    def combFunc(a, b):
        return a + b  # 累加起来

    print(' =====num:', num)
    sc = init_spark(' ')
    rdd = sc.parallelize([(1, 1), (1, 2), (2, 1), (2, 3), (2, 4), (1, 7)], num)
    aggregateRDD = rdd.aggregateByKey(3, seqFunc, combFunc)

    print('collect:', aggregateRDD.collect())  # tuple
    print('collectAsMap:', aggregateRDD.collectAsMap())  # dict


def aggregate_func_lambda(sc):
    rdd = sc.parallelize([('B', 1), ('B', 2), ('A', 3), ('A', 4), ('A', 5)])  # 默认被分为4个partition

    # 这个函数可以看见被分为几个partition
    def f(index, items):
        print("partitionId:%d" % index)
        for val in items:
            print(val)
        return items

    rdd.mapPartitionsWithIndex(f, False).count()

    zeroVal = 1
    mergeVal = (lambda aggregated, el: aggregated + el)  # aggregated即zeroVal
    mergeComb = (lambda agg1, agg2: agg1 + agg2)

    # aggregateByKey(self, zeroValue, seqFunc, combFunc,
    result = rdd.aggregateByKey(zeroVal, mergeVal, mergeComb)
    print(rdd.glom().collect())
    print(result.collect())


def addItemToList(lst, val):
    lst.append(val)
    return lst


def mergeLists(lst1, lst2):
    return lst1 + lst2


def merge_wrong():
    conf = SparkConf().setAppName("lg").setMaster('local[4]')
    sc1 = SparkContext.getOrCreate(conf)

    rdd = sc1.parallelize(
        [('a', 'ITEM_2', '2018'), ('b', 'USER_1', '2010'), ('a', 'ITEM_1', '2017'), ('a', 'ITEM_3', '2019'),
         ('b', 'UER', '2000')])  # type(rdd): <class 'pyspark.rdd.RDD'>

    rdd1 = rdd.sortBy(lambda x: x[2]).map(lambda x: (x[0], x[1]))
    print(rdd1.collect())
    zeroVal = list()
    mergeVal = (lambda aggregated, el: aggregated.append(el))  # 这个地方错误，可以用函数addItemToList替代
    mergeComb = (lambda agg1, agg2: agg1.extend(agg2))
    rdd2 = rdd.aggregateByKey(zeroVal, mergeVal, mergeComb)

    rdd = sc1.parallelize([('a', 'ITEM_2'), ('b', 'USER_1'), ('a', 'ITEM_1'), ('a', 'ITEM_3'), ('b', 'UER')],
                          1)  # type(rdd): <class 'pyspark.rdd.RDD'>


def merge_wright():
    sc = init_spark('myname')
    sql_str = 'select uid, id, ctime from recom.load_expose_click where pdate=20210521 and phour=00 and ' \
              'uid is not null and id is not null and ctime is not null and ' \
              'sch not in ("relateDocOrig", "relateVideoOrig")'
    rdd = sc.sql(sql_str).rdd
    rdd1 = rdd.sortBy(lambda x: x.ctime)  # 按ctime排序
    rdd2 = rdd1.map(lambda x: (x.uid, (x.id, x.ctime)))
    blank_list = []
    rdd3 = rdd2.aggregateByKey(blank_list, addItemToList, mergeLists)
    return rdd3


# target_partition = 'pdate=20210521/phour=01'
# next_partition = 'pdate=20210521/phour=02'
# expose_cmd = 'hive -e "show partitions recom.load_expose_click" 2>> hive_error |tail -300'
# profile_cmd = 'hive -e "show partitions rank.rank_news_profile_snapshot" 2>> hive_error |tail -300'
#
# date = '20210521'
# hour = '01'
# time_condition = 'pdate=' + date + ' and phour=' + hour
# sql_str = 'select uid, id, ctime from recom.load_expose_click where ' + time_condition + \
#           ' and uid is not null and id is not null and ctime is not null and ' \
#           'sch not in ("relateDocOrig", "relateVideoOrig")'
# rdd_hour = sc.sql(sql_str).rdd.map(lambda x: '\t'.join([x.uid, x.id, x.ctime]))
# save_path = '/user/recom/recall/mind/click_log/2021052101'
