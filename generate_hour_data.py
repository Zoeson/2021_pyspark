#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Time : 2021/6/5 15:08
# Author : 
# File : generate_hour_data.py
# Software: PyCharm
# description:
"""
    之前想着是每一小时都把一周的数据准备好，写入hdfs,然后模型每小时直接去hdfs上读数据，
    但是如果不能增量的进行数据读取，每次都往hdfs写入1周的数据，量太大，写不上
    只能，每次写到hdfs一小时的数据，然后，down到本机内存，然后看内存的处理量

0605尝试一：
    1. 写24个小时的数据到hdfs上，然后就够了
    2. 写代码，尝试连续处理这24个小时的数据，包括去重，去掉user小于6次点击
    MIND属于无序行为序列？没有会话的概念

0607:
    item 固定特征：
        category是 rank_news_profile_snapshot里，图文的一级分类，但是没有对应的subcate, 有classv1, classv2
    user 非固定特征：
        docpic_cate, video_cate，是用户表里，用户浏览过的文章的 视频/图文的 一级/二级 分类 （这些不是固有特征，先不加入）
"""
import csv
import os
import sys
import datetime
import time
import re
import findspark
findspark.init()

from pyspark.sql import SparkSession


def exec_cmd(cmd):
    r = os.popen(cmd)
    text = r.read()
    r.close()
    return text.strip('\n')


def init_spark():
    spark = SparkSession\
        .builder\
        .appName('name')\
        .enableHiveSupport()\
        .config('spark.driver.host',  'localhost')\
        .config('spark.sql.hive.convertMetastoreParquet', False)\
        .getOrCreate()
    return spark


def get_latest_date_hour():
    base_dir = "/user/recom/recall/mind/"

    date_cmd = 'hadoop fs -ls ' + base_dir + ' | cut -d "/" -f 6 | sort -n | tail -1'

    date = exec_cmd(date_cmd)

    # 返回最新hour
    date_dir = base_dir + '/' + date
    hour_cmd = 'hadoop fs -ls ' + date_dir + ' | cut -d "/" -f 7 |  grep -E "[0-9][0-9]" |sort -n | tail -1'
    hour = exec_cmd(hour_cmd)

    # 检查最新事件的文件夹是否有SUCCESS,如果成功，继续，否则，退出
    success_click_cmd = 'hadoop fs -test -e ' + base_dir + date + '/' + hour + '/click_log/_SUCCESS'
    sys_value_click = os.system(success_click_cmd)

    success_click_item = 'hadoop fs -test -e ' + base_dir + date + '/' + hour + '/item_profile/_SUCCESS'
    sys_value_item = os.system(success_click_item)

    success_click_user = 'hadoop fs -test -e ' + base_dir + date + '/' + hour + '/user_profile/_SUCCESS'
    sys_value_user = os.system(success_click_user)

    if sys_value_click == 0 and sys_value_item == 0 and sys_value_user == 0:
        print('_SUCCESS')
    else:
        print('!!! !! check latest DATA failed! not exist: click_log ? item_profile? user_profile?\n', success_click_cmd)
        sys.exit(-1)

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

    expose_cmd = 'hive -e "show partitions recom.load_expose_click" 2>>./hive_error |tail -300'
    profile_cmd = 'hive -e "show partitions rank.rank_news_profile_snapshot" 2>>./hive_error |tail -300'

    expose_result = exec_cmd(expose_cmd)
    profile_result = exec_cmd(profile_cmd)

    cond_log = (partition_target in expose_result)
    cond_profile = (partition_target.replace('pdate', 'pday') in profile_result) and \
                   (partition_next_hour.replace('pdate', 'pday') in profile_result)

    return cond_log and cond_profile


def get_hourly_click_log(sc, date, hour):
    """
    修改字段
    :param date:
    :param hour:
    :return:
    """
    save_path = '/user/recom/recall/mind/' + date + '/' + hour + '/click_log'
    print('SAVE click_log DATA: {}'.format(save_path))

    time_condition = 'pdate=' + date + ' and phour=' + hour
    sql_str = 'select uid, id, ctime from recom.load_expose_click where ' + time_condition + \
              ' and uid is not null and id is not null and ctime is not null and ' \
              'sch not in ("relateDocOrig", "relateVideoOrig")'

    rdd = sc.sql(sql_str)
    print('    RDD COLUMNS:   {}'.format(rdd.columns))
    rdd.repartition(10).write.format("csv").option("delimiter", "\t").save(save_path)


# ================================================================
def get_hourly_item_profile(sc, date, hour):
    """
    todo: 都需要添加哪些信息
    recom_id, doctype, simId, category, mediaId:这些特征不需要添加dumptime,固定特征
    distype, duration, sourcelevel, , qualitylevel, timeSensitiveLevel
    :param date:
    :param hour:
    :return:
    """
    save_path = '/user/recom/recall/mind/' + date + '/' + hour + '/item_profile'
    print('SAVE item_profile DATA: {}'.format(save_path))

    time_condition = 'pday=' + date + ' and phour=' + hour

    sql_str = "select recom_id, doctype, simId, get_json_object(regexp_replace(category,'\n|\t|\r', ''),'$[0]')," \
              " mediaId from rank.rank_news_profile_snapshot where " + time_condition + \
              " and recom_id is not null"

    # sql_str = "select recom_id, doctype, simId, regexp_replace(category,'\n|\t|\r', '')," \
    #           " mediaId from rank.rank_news_profile_snapshot where " + time_condition + \
    #           " and recom_id is not null"

    rdd = sc.sql(sql_str)
    print('    RDD COLUMNS:   {}'.format(rdd.columns))
    rdd.repartition(10).write.format("csv").option("delimiter", "\t").save(save_path)


def get_hourly_user_profile(sc, date, hour):
    """
    todo: rank.rank_user_profile_snapshot
    uid,
    ulevel
    suppose_gender, suppose_age,
    user_group,
    general_likevidr,
    general_doc_timesensitive,
    general_vid_timesensitive,
    general_timesensitive

    视频特征：general_likevidr_dayinwork, general_likevidr_dayinwork, general_likevidr_weekends，general_likevidr
    非固定特征：video_cate, docpic_cate,

    :param date:
    :param hour:
    :return:
    """
    save_path = '/user/recom/recall/mind/' + date + '/' + hour + '/user_profile'
    print('SAVE item_profile DATA: {}'.format(save_path))

    time_condition = 'pday=' + date + ' and phour=' + hour
    sql_str = 'select uid, ulevel,suppose_gender, suppose_age, user_group, general_likevidr, general_doc_timesensitive,' \
              ' general_vid_timesensitive, general_timesensitive from rank.rank_user_profile_snapshot where ' \
              + time_condition + ' and uid is not null'

    rdd = sc.sql(sql_str)
    print('    RDD COLUMNS:   {}'.format(rdd.columns))
    rdd.repartition(10).write.format("csv").option("delimiter", "\t").save(save_path)


def clean(max_keep=7):
    cmd = "hadoop fs -ls /user/recom/recall/mind/"   # click_log
    cmd_result = exec_cmd(cmd)
    pattern = '/user/recom/recall/mind/[0-9]{8}'
    dir_list = sorted(re.findall(pattern, cmd_result))
    length = len(dir_list)
    if length > max_keep:
        for i in range(0, length-max_keep):
            d = dir_list[i]
            print("deleting:", d)
            exec_cmd("hadoop fs -rm -r " + d)


def get_run_time():
    latest_date, latest_hour = get_latest_date_hour()
    print('==========DATA dir latest:{}'.format(latest_date + latest_hour))
    return get_new_time(latest_date, latest_hour, 1)


def get_new_time(date_str, hour_str, num):
    now_time_str = date_str + hour_str
    now_time = datetime.datetime.strptime(now_time_str, '%Y%m%d%H')
    delta = datetime.timedelta(hours=num)
    next_time = now_time + delta
    next_time_str = next_time.strftime('%Y%m%d%H')
    return next_time_str[:8], next_time_str[8:]


if __name__ == "__main__":
    sc = init_spark()

    run_date, run_hour = get_run_time()
    print('====run date hour: {}'.format(run_date+run_hour))

    while True:
        if check_data(run_date, run_hour):
            get_hourly_click_log(sc, run_date, run_hour)
            get_hourly_item_profile(sc, run_date, run_hour)
            get_hourly_user_profile(sc, run_date, run_hour)
            print("cleaning history data ...")
            clean(7)
            time.sleep(5)

        else:
            print("data not ready, time sleep 10 minutes")
            time.sleep(600)

        run_date, run_hour = get_run_time()