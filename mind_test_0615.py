#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Time : 2021/6/15 16:09
# Author : 
# File : mind_test_0615.py
# Software: PyCharm
# description:
"""
开始测试mind模型
"""

import pandas as pd
import random
import numpy as np
from sklearn.preprocessing import LabelEncoder
from model.feature_column import SparseFeat, VarLenSparseFeat
from model.mind import MIND
from tensorflow.python.keras import backend as K
from model.utils import sampledsoftmaxloss
from tensorflow.python.keras.models import Model
from process_data import preprocess_data, gen_data_set, gen_model_input
import numpy as np
import faiss
from tqdm import tqdm

# if __name__ == '__main__':
K.set_learning_phase(True)
import tensorflow as tf


def faiss_ivf_PQ(data, dim, m, nlist, nprobe=1):
    print('====================### IVF PQ ###==========================')
    quantizer = faiss.IndexFlatL2(dim)
    index_IVF = faiss.IndexIVFPQ(quantizer, dim, nlist, m, 8)
    index_IVF.train(data)  # index_IVF.is_trained=True， 建码本
    index_IVF.add(data)  # 添加索引
    index_IVF.nprobe = nprobe  # 默认nprobe=1
    return index_IVF


def faiss_ivf_FLAT(item_embs, embedding_dim, nlist):
    quantizer = faiss.IndexFlatL2(embedding_dim)  # the other index
    index = faiss.IndexIVFFlat(quantizer, embedding_dim, nlist, faiss.METRIC_L2)  # index_IVF.is_trained=Flase
    index.train(item_embs)  # index_IVF.is_trained=True
    index.add(item_embs)  # 建索引


def merge_faiss_result(distance, indexId):
    print(distance.shape)
    dis_list = distance.reshape(1, -1)[0]
    print('dis_list.shape:', len(dis_list))
    idx_list = indexId.reshape(1, -1)[0]
    result = {}
    for key, dis in zip(idx_list, dis_list):
        if key in result:
            result[key] = min(dis, result[key])
        else:
            result[key] = dis
    if len(idx_list) != len(result):
        print(idx_list)
    # 排序，按照values的大小排序
    result = sorted(result.items(), key=lambda x: x[1], reverse=False)
    return result


model_name = 'MIND'

if tf.__version__ >= '2.0.0':
    tf.compat.v1.disable_eager_execution()
# x, y, user_feature_columns, item_feature_columns, item_all = get_xy()

# 1. read_data
data = pd.read_csv('./data_files/expose.csv', delimiter='\t')
print('data.shape:{}'.format(data.shape))
print(data.head())
data = preprocess_data(data[:1000000], 3, 6)

# 2. Label Encoding for sparse features,
SEQ_LEN = 64
negsample = 0
features = ['uid', 'id']
feature_max_idx = {}

for feature in features:
    lbe = LabelEncoder()
    data[feature] = lbe.fit_transform(data[feature]) + 1
    feature_max_idx[feature] = data[feature].max() + 1

# 3. 配置一下模型定义需要的特征列，主要是特征名和embedding词表的大小
embedding_dim = 16
user_feature_columns = [SparseFeat('user_id', feature_max_idx['uid'], embedding_dim),
                        VarLenSparseFeat(SparseFeat('hist_item_id', feature_max_idx['id'], embedding_dim,
                                                    embedding_name="item_id"), SEQ_LEN, 'mean', 'hist_len'),
                        ]
item_feature_columns = [SparseFeat('item_id', feature_max_idx['id'], embedding_dim)]

# 4. train data and test data
train_set, test_set = gen_data_set(data, negsample)
train_model_input, train_label = gen_model_input(train_set, SEQ_LEN)
test_model_input, test_label = gen_model_input(test_set, SEQ_LEN)

# 5. init, compile, fit MIND model
model = MIND(user_feature_columns, item_feature_columns, num_sampled=5, user_dnn_hidden_units=(64, 16),
             dynamic_k=True)

model.compile(optimizer='adam', loss=sampledsoftmaxloss)
model.fit(train_model_input, train_label, batch_size=2048, epochs=2, validation_split=0.5)
print(model_name + " test train valid pass!")

# 6. user and item embedding model:  model.user_input=inputs_list  model.item_input=item_inputs_list
user_embedding_model = Model(inputs=model.user_input, outputs=model.user_embedding)
item_embedding_model = Model(inputs=model.item_input, outputs=model.item_embedding)

# 7. user and item data for embedding
test_user_model_input = test_model_input
# test_user_model_input = {"user_id": train_uid, "movie_id": train_iid, "hist_movie_id": train_seq_pad,
#                          "hist_len": train_hist_len}
item_profile = data[["id"]].drop_duplicates('id')
all_item_model_input = {"item_id": item_profile['id'].values}  # [[36608], [39277], [], ...]
print('all_item_model_input:', all_item_model_input['item_id'][:20])

user_embs = user_embedding_model.predict(test_user_model_input, batch_size=2 ** 12)
item_embs = item_embedding_model.predict(all_item_model_input, batch_size=2 ** 12)

print('##### USER_embs.shape:', user_embs.shape)
print('item_embs.shape:', item_embs.shape)

# 5. [Optional] ANN search by faiss  and evaluate the result
test_true_label = {line[0]: [line[2]] for line in test_set}  # uid: next_click_id

# index = faiss.IndexIVFFlat(embedding_dim)
nlist = 100
nprobe = 10
m = 4  # 可以调
top_k = 10

# index = faiss_ivf_FLAT(item_embs, embedding_dim, nlist)
index = faiss_ivf_PQ(item_embs, embedding_dim, m, nlist)

for user in user_embs[:2]:
    distance, idx = index.search(user, top_k)  # emb: 4个向量。<class 'numpy.ndarray'> == (4, 32)
    merge_result = merge_faiss_result(distance, idx)

# ascontiguousarray函数将一个内存不连续存储的数组转换为内存连续存储的数组，使得运行速度更快。
# 变换前后，user_embs的类型不变，<class 'numpy.ndarray'>, shape不变：(item_num, 2, 16)
# D, I = index.search(np.ascontiguousarray(user_embs), 50)

top_N_list = [5, 10, 20, 50, 100]
score_N = np.zeros(len(top_N_list))
# test_user_model_input:{'user_id':[d1,d2, ...], ''}
for i, uid in tqdm(enumerate(test_user_model_input['user_id'])):
    try:
        for top_N in range(len(top_N_list)):
            user_embedding = user_embs[i]
            distance, idx = index.search(user_embedding, top_k)  # emb: 4个向量。<class 'numpy.ndarray'> == (4, 32)
            merge_result = merge_faiss_result(distance, idx)
            pred = [item_profile['id'].values[x] for x in merge_result]

            if test_true_label[uid] in pred[:N]:
                score_N[n] += 1
    except:
        print(i)

print('mean_score:', score_N / i)  # i为test_user_emb的个数

# s = [[1, 1, 1], [2, 2, 2], [3, 3, 3]]
# np.mean(s, 1)  # array([1,2,3])
