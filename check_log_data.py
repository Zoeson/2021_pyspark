#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Time : 2021/5/31 14:14
# Author : 
# File : check_log_data.py
# Software: PyCharm
# description:
"""
根据目前文件夹的日期，检查最新数据是否更新
1. 现在文件夹的日期
参考 ffm_run_0807.py
    sid的作用？？

    0601：解决warn 的问题：
        21/06/01 10:55:59 WARN DFSClient: Failed to connect to /10.64.224.43:4001 for block, add to deadNodes and continue. java.nio.channels.ClosedByInterruptException
            java.nio.channels.ClosedByInterruptException


sql_str = 'select uid, id, ctime from recom.load_expose_click where pdate=20210521 and phour=00 and uid is not null ' \
      'and id is not null and ctime is not null and sch not in ("relateDocOrig", "relateVideoOrig")'

    date = '20210521'
    hour = '01'
    out_path = '/user/recom/recall/mind/click_log/20210521/01/hour_log'
    file_path = '/user/recom/recall/mind/click_log/20210521/01/hour_log'


set mapred.job.queue.name=recom;
set mapreduce.map.memory.mb=10150;
set mapreduce.map.java.opts=-Xmx6144m;
set mapreduce.reduce.memory.mb=10150;
set mapreduce.reduce.java.opts=-Xmx8120m;

sql_str = 'select count(*) from recom.load_expose_click where concat(pdate,phour) between "2021052101" and "2021052102" and uid is not null ' \
      'and id is not null and ctime is not null and sch not in ("relateDocOrig", "relateVideoOrig")'

"""
import os
import sys
import datetime
import time
import re
from pyspark.sql import SparkSession


def add_item2list(lst, val):
    lst.append(val)
    return lst


def merge_lists(lst1, lst2):
    return lst1 + lst2


def exec_cmd(cmd):
    r = os.popen(cmd)
    text = r.read()
    r.close()
    return text.strip('\n')


def get_latest_date_hour():
    base_dir = "/user/recom/recall/mind/click_log"
    date_cmd = 'hadoop fs -ls ' + base_dir + ' | cut -d "/" -f 7 | sort -n | tail -1'
    date = exec_cmd(date_cmd)

    # 返回最新hour
    date_dir = base_dir + '/' + date
    hour_cmd = 'hadoop fs -ls ' + date_dir + ' | cut -d "/" -f 8 | sort -n | tail -1'
    hour = exec_cmd(hour_cmd)

    # 检查最新事件的文件夹是否有SUCCESS,如果成功，继续，否则，退出
    success_path = base_dir + '/' + date + '/' + hour + '/_SUCCESS'
    success_cmd = 'hadoop fs -test -e ' + success_path
    sys_value = os.system(success_cmd)
    if sys_value != 0:
        print('!!! !!! check latest DATA failed! not exist: ', success_path)
        sys.exit(-1)
    else:
        print('_SUCCESS')
    return date, hour


def get_new_date_hour(date_str, hour_str, num):
    """
    把 date_hour合并，利用datetime计算时间
    :param date_str:
    :param hour_str:
    :param num:  正：往后num小时，负：往前num小时
    :return:

    """
    now_time = datetime.datetime.strptime(date_str + hour_str, '%Y%m%d%H')
    delta = datetime.timedelta(hours=num)
    next_time = now_time + delta
    next_time_str = next_time.strftime('%Y%m%d%H')
    return next_time_str[:8], next_time_str[8:]


def check_data(date, hour):
    """
    为什么 用户画像、内容画像的数据要比 曝光数据多一个小时
    :return:
    """
    date_next, hour_next = get_new_date_hour(date, hour, 1)
    partition_target = 'pdate=' + date + '/phour=' + hour
    partition_next_hour = 'pdate=' + date_next + '/phour=' + hour_next
    print('target_partition:{}\nnext hour partition:{}'.format(partition_target, partition_next_hour))

    expose_cmd = 'hive -e "show partitions recom.load_expose_click" 2>>.hive/error |tail -300'
    profile_cmd = 'hive -e "show partitions rank.rank_news_profile_snapshot" 2>>.hive/error |tail -300'

    expose_result = exec_cmd(expose_cmd)
    profile_result = exec_cmd(profile_cmd)

    cond_log = (partition_target in expose_result)
    cond_profile = (partition_target.replace('pdate', 'pday') in profile_result) and \
                   (partition_next_hour.replace('pdate', 'pday') in profile_result)

    return cond_log and cond_profile


# ================================================================
def get_hourly_click_log(sc, date, hour):
    """
    :param date:
    :param hour:
    :return:
    """

    time_condition = 'pdate=' + date + ' and phour=' + hour
    sql_str = 'select uid, id, ctime from recom.load_expose_click where ' + time_condition + \
              ' and uid is not null and id is not null and ctime is not null and ' \
              'sch not in ("relateDocOrig", "relateVideoOrig")'

    rdd = sc.sql(sql_str).rdd.map(lambda x: [x[0], x[1], x[2]])
    # rdd1 = rdd.map(lambda x: '\t'.join([x[0], x[1], x[2]]))  # x.uid, x.id, x.ctime
    # out_path = '/user/recom/recall/mind/click_log/' + date + '/' + hour + '/hour_log'
    # rdd1.repartition(10).saveAsTextFile(out_path)
    return rdd


def read_hdfs(sc, date, hour):
    file_path = '/user/recom/recall/mind/click_log/' + date + '/' + hour + '/hour_log'

    rdd = sc.sparkContext.textFile(file_path)
    rdd_split = rdd.map(lambda line: line.split('\t'))
    return rdd_split


def init_spark():
    spark = SparkSession \
        .builder \
        .appName('name') \
        .enableHiveSupport() \
        .config("spark.driver.host", "localhost") \
        .getOrCreate()
    return spark


def get_weekly_data(sc, date, hour):
    """
    select max(int(ctime)), min(int(ctime)) from recom.load_expose_click where pdate=20210521 and phour=01 and ctime is not null;
    max:1621620600
    min:1621461702

    小时对应的时间戳
    t0 = "2021052100"   1621526400
    t1 = "2021052101"   1621530000
    t1 = "2021052102"   1621533600

    select count(*) from recom.load_expose_click where pdate=20210521 and phour=01 and ctime is not null;
    total不为空：261741

    select count(*) from recom.load_expose_click where pdate=20210521 and phour=01 and int(ctime)< 1621533600 and int(ctime) >1621530000;
    t0 -t1 : 402
    t1 - t2: 235325
    ====> phour=01 是01点-02点之间的数据

    下面两个语句，结果一样
    select count(*) from recom.load_expose_click where pdate=20210521 and phour=01 and ctime >'1621530000' and ctime< '1621533600';
    select count(*) from recom.load_expose_click where pdate=20210521 and phour=01 and int(ctime) >1621530000 and int(ctime) < 1621533600;
    =====================================
    select count(*) from recom.load_expose_click where pdate=20210521 and phour=01 and ctime is not null;
    total不为空：261741
    select count(*) from recom.load_expose_click where pdate=20210521 and phour=01 and ctime >'1621530000' and ctime< '1621533600';
    01-02：235325
    select count(*) from recom.load_expose_click where pdate=20210521 and phour=01 and ctime is not null and ctime <='1621530000';
    01点之前：7960
    select count(*) from recom.load_expose_click where pdate=20210521 and phour=01 and ctime is not null and ctime >='1621533600';
    02点之后：18456

    ======uid is not null and id is not null====
    select count(*) from recom.load_expose_click where pdate=20210521 and phour=01 and ctime is not null and uid is not null and id is not null and sch not in ("relateDocOrig", "relateVideoOrig");

    """

    date_before, hour_before = get_new_date_hour(date, hour, -1)
    read_path = '/user/recom/recall/mind/click_log/' + date_before + hour_before

    time_cond = 'pdate=' + date_before + ' and phour=' + hour_before
    print('=====Read WEEK LOG data: {}'.format(time_cond))

    #
    rdd_week = sc.sparkContext.textFile(read_path)
    rdd_split = rdd_week.map(lambda x: x.split('\t'))
    time_filter = int(time.mktime(time.strptime(date+hour, '%Y%m%d%H'))) - 5*60*60*1000
    rdd_filter = rdd_split.filter(lambda x: x[2] < time_filter)  # date=20210521 01: 1621530000

    print('rdd_split.count(): {} rdd_week.count(): {}\nrdd_filter.take(2): {}'.format(
        rdd_week.count(), rdd_filter.count(), rdd_filter.take(2)))

    # 新的小时的click数据
    rdd_hour = get_hourly_click_log(sc, date, hour)
    time_condition = 'pdate=' + date + ' and phour=' + hour
    print('====NEW HOUR LOG time: {}  rdd.count(): {} \nrdd.take(2): {}'.format(time_condition, rdd_hour.count(), rdd_hour.take(2)))

    rdd_week_new = rdd_filter.union(rdd_hour)
    print('===== MERGE ===')
    print('rdd_week_new.count(): {},  rdd_week_new.take(2):{}'.format(rdd_week_new.count(), rdd_week_new.take(2)))

    save_path = '/user/recom/recall/mind/click_log/' + date + hour
    rdd_week_new.repartition(100).saveAsTextFile(save_path)


def clean(datadir, maxKeep):
    cmd = "hadoop fs -ls /user/recom/recall/mind/" + datadir  # click_log
    cmd_result = exec_cmd(cmd)
    pattern = '/user/recom/recall/mind/'+datadir+'/[0-9]{8}'
    dir_list = sorted(re.findall(pattern, cmd_result))
    length = len(dir_list)
    if length > maxKeep:
        for i in range(0, length-maxKeep):
            d = dir_list[i]
            print("deleting:", d)
            exec_cmd("hadoop fs -rm -r " + d)


if __name__ == "__main__":
    # 检查现在文件的最新日期
    sc = init_spark()
    latestDate, latestHour = get_latest_date_hour()
    # 返回接下来要拿数据的时间
    next_date, next_hour = get_new_date_hour(latestDate, latestHour, 1)
    # 检查数据是否准备好
    while True:
        if check_data():
            get_weekly_data(sc, next_date, next_hour)  # 新数据，合并老数据，写入week_log
            print("cleaning history data ...")
            clean('click_log', 6)
            time.sleep(5)

        else:
            print("data not ready, time sleep 10 minutes")
            time.sleep(600)
