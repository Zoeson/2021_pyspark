#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Time : 2021/5/28 20:58
# Author : 
# File : test_pyspark.py
# Software: PyCharm
# description:
"""
第一个spark程序:
    如果想读hive表，必须在服务器上运行，
"""
from pyspark.sql import SparkSession

import pyspark
from pyspark import SparkContext
from pyspark import SparkConf


def init_spark(name):
    spark = SparkSession \
        .builder \
        .appName(name) \
        .enableHiveSupport() \
        .config("spark.driver.host", "localhost") \
        .getOrCreate()
    return spark


def spark_sql():
    """
    spark sql：末尾不要加";"
    :return:
    """
    # 这样读hive不太好使，读到的data是乱码，并且是Row()这种格式，不好处理
    # data_path = '/user/hive/warehouse/rank.db/rank_news_profile_snapshot/pday=20210528/phour=20/'
    # df = sc.read.text(data_path)

    sc = init_spark('myname')
    sql_str = 'select recom_id, simid, title from rank.rank_news_profile_snapshot where pday=20210528 and phour = 00 limit 10'
    df = sc.sql(sql_str)  # <class 'pyspark.sql.dataframe.DataFrame'>
    df_rdd = df.rdd  #
    rdd = df_rdd.map(lambda line: '\t'.join([line.recom_id, line.simid, line.title]))
    print("example:", rdd.take(2))
    print('count:', rdd.count())  # 10

    # rdd存入hdfs
    out_path = 'user/recom/recall/mind/spark_tmp/'
    rdd.repartition(20).saveAsTextFile(out_path)


def add(a, b):
    c = a + b
    return c


def reduce_group(sc):
    rdd = sc.parallelize(
        [('a', 3), ('b', 1), ('a', 1), ('a', 1), ('b', 1), ('b', 1)])
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


# if __name__ == "__main__":
from pyspark import SparkContext
from pyspark import SparkConf





