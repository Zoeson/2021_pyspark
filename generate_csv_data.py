#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Time : 2021/6/9 16:33
# Author : 
# File : generate_csv_data.py
# Software: PyCharm
# description:
"""
每小时的数据。保存到csv
新小时的数据，合并，写入到新小时的csv
20210521 01, 一个小时的数据是7.1M, 假设平均10M,一天240M,7天共1.68G

ps:
    时间戳加减，time.mktime = hour_num*60*60,没有1000!没有1000!没有1000!,或者直接计算hour_num小时之前日期,转化时间戳，不要这样减

"""

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
        .config('spark.driver.host', 'localhost') \
        .config('spark.sql.hive.convertMetastoreParquet', False) \
        .getOrCreate()
    return spark


def exec_cmd(cmd):
    r = os.popen(cmd)
    text = r.read()
    r.close()
    return text.strip('\n')


def get_new_time(date, hour, num):
    now_time = datetime.datetime.strptime(date + hour, '%Y%m%d%H')
    delta = datetime.timedelta(hours=num)
    next_time = now_time + delta
    next_time_str = next_time.strftime('%Y%m%d%H')
    return next_time_str[:8], next_time_str[8:]


def check_data_ready(date, hour):
    partition_target = 'pdate=' + date + '/phour=' + hour  # log data: contain run_hour

    date_next, hour_next = get_new_time(date, hour, 1)
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

    date = exec_cmd(date_cmd)

    # 返回最新hour
    date_dir = base_dir + '/' + date
    hour_cmd = 'hadoop fs -ls ' + date_dir + ' | cut -d "/" -f 8 |  grep -E "[0-9][0-9]" |sort -n | tail -1'
    hour = exec_cmd(hour_cmd)

    # 检查最新事件的文件夹是否有SUCCESS,如果成功，继续，否则，退出
    success_click_cmd = 'hadoop fs -test -e ' + base_dir + '/' + date + '/' + hour + '/_SUCCESS'
    sys_value_click = os.system(success_click_cmd)

    if sys_value_click == 0:
        print('_SUCCESS')
    else:
        print('!!! !! check latest DATA failed! not exist: click_log', success_click_cmd)
        sys.exit(-1)

    return date, hour


def get_run_time():
    latest_date, latest_hour = get_latest_date_hour()
    print('==========DATA dir latest:{}'.format(latest_date + latest_hour))
    new_data, new_hour = get_new_time(latest_date, latest_hour, 1)
    return latest_date, latest_hour, new_data, new_hour


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


def get_new_data(sc, date, hour, next_date, next_hour, hour_num):
    """
        读上一小时的csv
        过滤掉大于1天的数据
        合并新的数据，
        写入新时刻的csv
    :param sc:
    :param date:
    :param hour:
    :param next_date:
    :param next_hour:
    :return:
    """
    read_path = '/user/recom/recall/mind/click_log/' + date + '/' + hour
    output_path = '/user/recom/recall/mind/click_log/' + next_date + '/' + next_hour

    # read csv:
    df = sc.read.format('csv').option('sep', '\t').option('header', 'true').csv(read_path)

    # filter
    filter_date, filter_hour = get_new_time(next_date, next_hour, -1 * hour_num)
    print('run time:{},  filter {} hours'.format(next_date+next_hour, hour_num))
    time_filter = int(time.mktime(time.strptime(filter_date + filter_hour, '%Y%m%d%H')))
    df_filter = df.filter(df['ctime'] > time_filter)  # date=20210521 01: 1621530000
    # print('ori.count:', df.count(), '   after filter count:', df_filter.count())

    # read new hour data. dropDuplicates
    time_condition = 'pdate=' + next_date + ' and phour=' + next_hour
    sql_str = 'select uid, id, ctime from recom.load_expose_click where ' + time_condition + \
              ' and uid is not null and id is not null and ctime is not null and ctime > 0' \
              ' and sch not in ("relateDocOrig", "relateVideoOrig") and (dur_sec >=10.0 or readrate >= 10.0)' \
              ' and itemtype in ("docpic", "video", "minivideo")'
    rdd_hour = sc.sql(sql_str)
    print('NEW DATA READ FINIESH')
    data_all = df_filter.union(rdd_hour)
    print('UNION FINISHED')
    data_all = data_all.dropDuplicates(['uid', 'id'])
    print('drop dupli FINISHED')

    # write csv
    t = datetime.datetime.now()
    data_all.repartition(10).write.format("csv").option("delimiter", "\t").option('header', 'true').save(output_path)
    print('save consume time:{}'.format(datetime.datetime.now() - t))
    print('    count after union:', data_all.count())
    print('save + count consume time:{}'.format(datetime.datetime.now() - t))


if __name__ == "__main__":
    s_c = init_spark()
    date, hour, next_date, next_hour = get_run_time()

    while True:
        if check_data_ready(next_date, next_hour):
            print('========start time:{}'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
            # todo: 最终用1周的数据，但是先用一天的数据进行测试
            get_new_data(s_c, date, hour, next_date, next_hour, 24)

            print("cleaning history data ...")
            clean('click_log', 7)
            time.sleep(5)

        else:
            print("data not ready, time sleep 10 minutes")
            time.sleep(600)

        date, hour, next_date, next_hour = get_run_time()
