#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Time : 2021/6/9 14:56
# Author : 
# File : process_data.py
# Software: PyCharm
# description:
"""
    预处理数据：
        linux上一周全量数据进行测试时，item>=3 和 user>=6基本上去掉一半，
        preprocess_data(data, 3, 6)
        ====data.shape:(64605761, 3)  ===inum:3 ====unum6
        item过滤前后    ： 33 4643 | 18 6129
        data过滤item前后: 6460 5761 | 6441 4578
        user过滤前后    ： 311 7082 | 163 3194
        data过滤user    : 6441 4578| 6092 4125

    生成样本
    训练模型
    测试
"""

import pandas as pd
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import random
import numpy as np
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

from model.feature_column import SparseFeat, VarLenSparseFeat


def preprocess_data(data, inum=3, unum=6):
    """
        用户行为日志数据处理：
        iid<3     33.5万--》 18.6万
        uid<6     311.7万 --》 163.3万
        总click记录： 6460万 --》 6092万
    :param data:
    :return:
    """
    print('data shape:', data.shape)

    # item_id
    item_id_list = data['id']
    item_dict = Counter(item_id_list)
    item_val = [item_id for item_id, v in item_dict.items() if v >= inum]  # 21864
    data_fiter_item = data[data['id'].isin(item_val)]
    print('item过滤前后    ：', len(item_dict), '|', len(item_val))
    print('data过滤item前后:', data.shape, '|', data_fiter_item.shape)  # (103312, 6)

    # user
    user_id_list = data_fiter_item['uid']
    user_dict = Counter(user_id_list)
    user_val = [user_id for user_id, v in user_dict.items() if v >= unum]
    data_filter = data_fiter_item[data_fiter_item['uid'].isin(user_val)]
    print('==================item >={}, user>={} 之后的数据情况================'.format(inum, unum))
    print('user过滤前后    :', len(user_dict), "|", len(user_val))
    print('item过滤    :', len(data_fiter_item['id'].unique()), '|', len(data_filter['id'].unique()))
    print('data过滤之后    :', data_fiter_item.shape, '|', data_filter.shape)

    data_sort = data_filter.sort_values('ctime', ascending=True)

    return data_sort


def gen_data_set(data, negsample=0):
    print("sort_data:\n", data.head())
    train_set = []
    test_set = []
    for reviewerID, hist_data in tqdm(data.groupby('uid')):
        pos_list = hist_data['id'].tolist()

        for i in range(1, len(pos_list)):
            hist = pos_list[:i]
            if i != len(pos_list) - 1:
                train_set.append((reviewerID, hist[::-1], pos_list[i], 1, len(hist[::-1]), i))
            else:
                test_set.append((reviewerID, hist[::-1], pos_list[i], 1, len(hist[::-1]), i))

    random.shuffle(train_set)
    random.shuffle(test_set)

    print(len(train_set[0]), len(test_set[0]))

    return train_set, test_set


def gen_model_input(train_set, seq_max_len):
    train_uid = np.array([line[0] for line in train_set])
    train_seq = [line[1] for line in train_set]
    train_iid = np.array([line[2] for line in train_set])
    train_label = np.array([line[3] for line in train_set])
    train_hist_len = np.array([line[4] for line in train_set])

    # 填补缺失值
    train_seq_pad = pad_sequences(train_seq, maxlen=seq_max_len, padding='post', truncating='post', value=0)
    train_model_input = {"user_id": train_uid, "item_id": train_iid, "hist_item_id": train_seq_pad,
                         "hist_len": train_hist_len}

    return train_model_input, train_label


def get_xy():
    # 1. read_data
    data = pd.read_csv('./data_files/expose.csv', delimiter='\t')
    print('data.shape:{}'.format(data.shape))
    print(data.head())
    data = preprocess_data(data[:1000000])

    # 2. Label Encoding for sparse features,
    SEQ_LEN = 64
    negsample = 0
    features = ['uid', 'id']
    feature_max_idx = {}

    for feature in features:
        lbe = LabelEncoder()
        data[feature] = lbe.fit_transform(data[feature]) + 1
        feature_max_idx[feature] = data[feature].max() + 1

    train_set, test_set = gen_data_set(data, negsample)
    train_model_input, train_label = gen_model_input(train_set, SEQ_LEN)
    test_model_input, test_label = gen_model_input(test_set, SEQ_LEN)

    # 3. 配置一下模型定义需要的特征列，主要是特征名和embedding词表的大小
    embedding_dim = 16
    user_feature_columns = [SparseFeat('user_id', feature_max_idx['uid'], embedding_dim),

                            VarLenSparseFeat(SparseFeat('hist_item_id', feature_max_idx['id'], embedding_dim,
                                                        embedding_name="item_id"), SEQ_LEN, 'mean', 'hist_len'),
                            ]
    item_feature_columns = [SparseFeat('item_id', feature_max_idx['id'], embedding_dim)]

    item_unique = data['id'].uinique()

    return train_model_input, train_label, user_feature_columns, item_feature_columns, item_unique


import findspark
findspark.init()
from pyspark.sql import SparkSession


def init_spark():
    """
    服务器 client提交，有下面2行会报错
    .config('spark.sql.hive.convertMetastoreParquet', False) \
    .config('spark.driver.host', 'localhost') \

    .appName('name')： 这一行是用yarn提交时，显示的spark的name
    :return:
    """
    spark = SparkSession \
        .builder \
        .appName('name') \
        .enableHiveSupport() \
        .getOrCreate()
    return spark


def test_week_data():
    sc = init_spark()
    date = '20210611'
    hour = '08'
    read_path = '/user/recom/recall/mind/click_log/' + date + '/' + hour
    data = sc.read.format('csv').option('sep', '\t').option('header', 'true').csv(read_path)
    preprocess_data(data)




