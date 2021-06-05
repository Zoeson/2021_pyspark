#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Time : 2021/6/4 18:35
# Author : 
# File : sql_1d.py
# Software: PyCharm
# description:


import os
import sys
import datetime
import time
import re
import findspark
findspark.init()

from pyspark.sql import SparkSession


def init_spark():
    spark = SparkSession \
        .builder \
        .appName('name') \
        .enableHiveSupport() \
        .getOrCreate()
    return spark


def exec_cmd(cmd):
    r = os.popen(cmd)
    text = r.read()
    r.close()
    return text.strip('\n')


def get_new_time(now_time_str, num):
    now_time = datetime.datetime.strptime(now_time_str, '%Y%m%d%H')
    delta = datetime.timedelta(hours=num)
    next_time = now_time + delta
    next_time_str = next_time.strftime('%Y%m%d%H')
    return next_time_str[:8], next_time_str[8:]


def check_data_ready(date, hour):
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


def get_latest_date_hour():
    base_dir = "/user/recom/recall/mind/click_log"

    date_cmd = 'hadoop fs -ls ' + base_dir + ' | cut -d "/" -f 7 | sort -n | tail -1'
    latest_date = exec_cmd(date_cmd)    # date latest
    assert len(latest_date) == 10
    print('======latest_date:{}'.format(latest_date))

    success_path = base_dir + '/' + latest_date + '/_SUCCESS'    # check: SUCCESS
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
    next_date, next_hour = get_new_time(latest_date, 1)
    print('             RUN date hour: {}'.format(next_date + next_hour))
    return next_date, next_hour


def clean(data_dir, max_keep):
    cmd = "hadoop fs -ls /user/recom/recall/mind/" + data_dir  # click_log
    cmd_result = exec_cmd(cmd)
    pattern = '/user/recom/recall/mind/' + data_dir + '/[0-9]{10}'
    dir_list = sorted(re.findall(pattern, cmd_result))
    length = len(dir_list)
    if length > max_keep:
        for i in range(0, length - max_keep):
            d = dir_list[i]
            print("deleting:", d)
            exec_cmd("hadoop fs -rm -r " + d)


def get_weekly_data(sc,  date, hour):
    """
    :param sc:
    :param date:
    :param hour:
    :return:
    """

    # date = '20210521'
    # hour = '00'
    now_time = date + hour
    output_path = '/user/recom/recall/mind/click_log/' + now_time
    sql_date, sql_hour = get_new_time(now_time, -1 * 24)   # 一周前
    sql_start = sql_date + sql_hour
    print('         now_time:{}, \n  data_start_time:{}'.format(now_time, sql_start))

sql_str = 'select uid, id, ctime from recom.load_expose_click where concat(pdate, phour) between {} and {} and ' \
          'uid is not null and id is not null and ctime is not null and sch not in ("relateDocOrig", "relateVideoOrig")'. \
    format(sql_start, now_time)

rdd_data = sc.sql(sql_str).rdd.map(lambda line: '\t'.join([line.uid, line.id, line.ctime]))

sc = init_spark()
sql_item = 'select distinct(uid) from rank.rank_user_profile_snapshot where concat(pday,phour) between "2021052101" and "2021052201" and uid is not null and dump_time is not null and recom_time is not null'
sql_user = 'select distinct(recom_id) from rank.rank_news_profile_snapshot where concat(pday,phour) between "2021052101" and "2021052201" and recom_id is not null and dump_time is not null and recom_time is not null'

rdd_item = sc.sql(sql_item).rdd.groupByKey().mapValues(len)
rdd_filter = rdd_item.filter(lambda x:x[1] >3)


if __name__ == "__main__":
    s_c = init_spark()

    run_date, run_hour = get_run_time()

    while True:
        if check_data_ready(run_date, run_hour):
            print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&& data ready &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
            print('========start time:{}'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
            get_weekly_data(s_c, run_date, run_hour)

            print("cleaning history data ...")
            clean('click_log', 10)
            time.sleep(5)

        else:
            print("data not ready, time sleep 10 minutes")
            time.sleep(600)

        run_date, run_hour = get_run_time()


