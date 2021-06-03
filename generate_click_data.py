#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Time : 2021/6/2 17:56
# Author : 
# File : generate_click_data.py
# Software: PyCharm
# description:
import os
import sys
import datetime
import time
import re
from pyspark.sql import SparkSession


def init_spark():
    spark = SparkSession \
        .builder \
        .appName('name') \
        .enableHiveSupport() \
        .config("spark.driver.host", "localhost") \
        .getOrCreate()
    return spark


def exec_cmd(cmd):
    r = os.popen(cmd)
    text = r.read()
    r.close()
    return text.strip('\n')


def get_latest_date_hour():
    base_dir = "/user/recom/recall/mind/click_log"

    # date latest
    date_cmd = 'hadoop fs -ls ' + base_dir + ' | cut -d "/" -f 7 | sort -n | tail -1'
    latest_date = exec_cmd(date_cmd)

    print('======latest_date:{}'.format(latest_date))
    # check: SUCCESS
    success_path = base_dir + '/' + latest_date + '/_SUCCESS'
    success_cmd = 'hadoop fs -test -e ' + success_path
    sys_value = os.system(success_cmd)
    if sys_value != 0:
        print('!!! !!! check latest DATA failed! not exist: ', success_path)
        sys.exit(-1)
    else:
        print('_SUCCESS')

    return latest_date


def get_run_time():
    latest_date = get_latest_date_hour()
    print('==========DATA dir latest:{}'.format(latest_date))
    return get_new_time(latest_date, 1)


def get_new_time(now_time_str, num):
    now_time = datetime.datetime.strptime(now_time_str, '%Y%m%d%H')
    delta = datetime.timedelta(hours=num)
    next_time = now_time + delta
    next_time_str = next_time.strftime('%Y%m%d%H')
    return next_time_str[:8], next_time_str[8:]


def check_data(date, hour):
    partition_target = 'pdate=' + date + '/phour=' + hour  # log data: contain run_hour

    date_next, hour_next = get_new_time(date+hour, 1)
    partition_next_hour = 'pdate=' + date_next + '/phour=' + hour_next  # user,item：contain hour after run_hour
    print('target_partition:{}\nnext_partition:{}'.format(partition_target, partition_next_hour))

    expose_cmd = 'hive -e "show partitions recom.load_expose_click" 2>> hive_error |tail -500'
    profile_cmd = 'hive -e "show partitions rank.rank_news_profile_snapshot" 2>> hive_error |tail -500'

    expose_result = exec_cmd(expose_cmd)
    profile_result = exec_cmd(profile_cmd)

    cond_log = (partition_target in expose_result)
    cond_profile = (partition_target.replace('pdate', 'pday') in profile_result) and \
                   (partition_next_hour.replace('pdate', 'pday') in profile_result)

    return cond_log and cond_profile


def get_weekly_data(sc,  date, hour):
    """
    todo: filter_condition: timeStamp
    rdd_filter = rdd_split.filter(lambda x: x[2] < '1621530000')  date=20210521 01: 1621530000
    :param sc:
    :param date_before:
    :param hour_before:
    :param date:
    :param hour:
    :return:
    """
    date_before, hour_before = get_new_time(date+hour, -1)
    time_cond = 'pdate=' + date_before + ' and phour=' + hour_before
    print('=====Read WEEK LOG data: {}'.format(time_cond))
    # read_path = '/user/recom/recall/mind/click_log/2021052100'
    read_path = '/user/recom/recall/mind/click_log/' + date_before + hour_before
    rdd_week = sc.sparkContext.textFile(read_path)
    rdd_split = rdd_week.map(lambda x: x.split('\t'))
    time_filter = str(int(time.mktime(time.strptime(date + hour, '%Y%m%d%H'))) - 5 * 60 * 60 * 1000)
    rdd_filter = rdd_split.filter(lambda x: x[2] > time_filter)
    print('rdd_split.count(): {} rdd_week.count(): {}\nrdd_filter.take(2): {}'.format(
        rdd_week.count(), rdd_filter.count(), rdd_filter.take(2)))

    # run_hour log
    time_condition = 'pdate=' + date + ' and phour=' + hour
    sql_str = 'select uid, id, ctime from recom.load_expose_click where ' + time_condition + \
              ' and uid is not null and id is not null and ctime is not null and ' \
              'sch not in ("relateDocOrig", "relateVideoOrig")'
    rdd_hour = sc.sql(sql_str).rdd.map(lambda x: [x.uid, x.id, x.ctime])
    print('====NEW HOUR LOG time: {}  rdd.count(): {} \nrdd.take(2): {}'.format(time_condition, rdd_hour.count(),
                                                                                rdd_hour.take(2)))

    rdd_week_new = rdd_filter.union(rdd_hour)
    print('===== MERGE ===')
    print('rdd_week_new.count(): {},  rdd_week_new.take(2):{}'.format(rdd_week_new.count(), rdd_week_new.take(2)))

    save_path = '/user/recom/recall/mind/click_log/' + date + hour
    rdd_week_new.map(lambda line: '\t'.join([x for x in line])).repartition(100).saveAsTextFile(save_path)


def clean(data_dir, max_keep):
    cmd = "hadoop fs -ls /user/recom/recall/mind/" + data_dir  # click_log
    cmd_result = exec_cmd(cmd)
    pattern = '/user/recom/recall/mind/' + data_dir + '/[0-9]{8}'
    dir_list = sorted(re.findall(pattern, cmd_result))
    length = len(dir_list)
    if length > max_keep:
        for i in range(0, length - max_keep):
            d = dir_list[i]
            print("deleting:", d)
            exec_cmd("hadoop fs -rm -r " + d)


if __name__ == "__main__":
    s_c = init_spark()

    run_date, run_hour = get_run_time()
    print('====run date hour: {}'.format(run_date+run_hour))

    while True:
        if check_data(run_date, run_hour):
            print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&& data ready &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
            print('start time:{}'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
            get_weekly_data(s_c, run_date, run_hour)
            print("cleaning history data ...")
            clean('click_log', 10)
            time.sleep(5)

        else:
            print("data not ready, time sleep 10 minutes")
            time.sleep(600)

        run_date, run_hour = get_run_time()
