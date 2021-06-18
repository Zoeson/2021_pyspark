#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Time : 2021/6/17 15:34
# Author : 
# File : test_faiss.py
# Software: PyCharm
# description:
"""
PQ，IVF详解: https://yongyuan.name/blog/ann-search.html
安装faiss, 在cmd下， 切到tf2这个虚拟环境，安装成功
    conda install faiss-cpu -c pytorch (windows安装faiss同样步骤)
faiss方法：
    FLAT和IVFFLAT都存储完整的向量
    1. faiss.IndexFlatL2  (全局暴力搜索)
        最简单的版本：IndexFlatL2，它只是对向量执行强力的L2距离搜索(暴力搜索, brute-force)。
    2. faiss.IndexIVFFLAT  （聚类）
    （分割为不同的单元，一个（或者几个）单元内搜索）
        精度损失主要是聚类带来的
        在d维空间中定义Voronoi单元格，并且每个数据库矢量都落入其中一个单元格中。
        在搜索时，只有查询x所在单元中包含的数据库向量(划分搜索空间)
        这IndexIVFFlat还需要另一个索引，即量化器(quantizer)，它将矢量分配给Voronoi单元
        参数：
            nlist 划分单元的数量
            nprobe 执行搜索访问的单元格数(不包括nlist)
    3. faiss.IndexIVFPQ  (聚类+PQ product quantizer)
        主要步骤：先Kmeans聚类，再用PQ（划分M个子空间，对每个子空间聚类n个簇），
        通过用不同子空间的质心ID 代表 数据库的向量（用于推荐系统，则指item的向量）对应子空间向量的方法，实现了对向量的压缩
        由于不需要直接存储向量数据，极大的减小了内存消耗，只需要存储：
            一个码本（质心ID:质心向量（比如128维划分为4个子空间，则每个子空间的质心是一个32维的向量）），
            数据库向量的编码结果（子空间向量用离它最近的质心代表, 只需要存质心ID）

        实验结果：
          最近的邻居被正确地找到（它是矢量ID本身），但是向量自身的估计距离不是0，这是由于：
            计算的是向量到质心的距离，而不是向量到向量的距离。
          另外搜索真实查询时，虽然结果大多是错误的(与FLATL2或者IVFFlat进行比较)，但是对于真实数据，效果反而更好，因为：
                统一数据很难进行索引，因为没有规律性可以被利用来聚集或降低维度(所以只用到聚类的 IVFFLAT  效果不好)
                对于自然数据，语义最近邻居往往比不相关的结果更接近。


"""

import numpy as np
import faiss


#######################
#  1. IndexFlatL2
#######################
def index_flat(data, dim, top_k):
    """
    64
    [[  0 393 363  78]
     [  1 555 277 364]
     [  2 304 101  13]
     [  3 173  18 182]
     [  4 288 370 531]]
    [[0.        7.175174  7.2076287 7.251163 ]
     [0.        6.323565  6.684582  6.799944 ]
     [0.        5.7964087 6.3917365 7.2815127]
     [0.        7.277905  7.5279875 7.6628447]
     [0.        6.763804  7.295122  7.368814 ]]
    """
    print('====================### FLAT 全局暴力搜索 ###==========================')
    index = faiss.IndexFlatL2(dim)  # 构建FlatL2索引
    print(index.is_trained)  # TRUE
    index.add(data)  # 向索引中添加向量
    print(index.ntotal)  # data的个数

    D, I = index.search(data[:5], top_k)  # 测试
    print(I)
    print(D)
    return index, D, I


#######################
#  2. IndexIVFFlat  : 将数据集分割成几部分。
# here we specify METRIC_L2, by default it performs inner-product search
#######################
def ivf_flat(data, dim, top_k, nlist):
    """
    :param data:
    :param dim:
    :param top_k: 最近邻的个数
    :param nlist: 划分单元的数量
    :return:
    """
    print('====================### IVF FLAT ###==========================')
    quantizer = faiss.IndexFlatL2(dim)  # the other index
    index_IVF = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_L2)  # index_IVF.is_trained=Flase
    index_IVF.train(data)  # index_IVF.is_trained=True
    index_IVF.add(data)  # 添加索引
    D1, I1 = index_IVF.search(data[:5], top_k)  # 搜索
    print(I1)
    print(D1)

    index_IVF.nprobe = 10  # 默认 nprobe 是1 ,可以设置的大一些试试
    D2, I2 = index_IVF.search(data[:5], top_k)  # 搜索
    print(I2)
    print(D2)
    return index_IVF, D2, I2


#######################
#  3. IndexIVFPQ
#######################
def ivf_pq(data, dim, top_k, nlist, m):
    """
    :param data:
    :param dim:
    :param top_k:
    :param nlist:   划分单元的数量
    :param m:  划分的子空间数： 128划分为4个子空间：Kmeans聚为256个簇
    :return:
    """
    print('====================### IVF PQ ###==========================')
    quantizer = faiss.IndexFlatL2(dim)  # the other index
    index_IVF = faiss.IndexIVFPQ(quantizer, dim, nlist, m, 8)  # 8表示编码为8个字节。index_IVF.is_trained=Flase
    index_IVF.train(data)  # index_IVF.is_trained=True， 建码本
    index_IVF.add(data)  # 添加索引
    index_IVF.nprobe = 10  # 默认nprobe=1
    D, I = index_IVF.search(data[:5], top_k)  # 搜索
    print(I)
    print(D)
    return D, I


def ivf_PQ(data, dim, m, nlist, nprobe=1):
    print('====================### IVF PQ ###==========================')
    quantizer = faiss.IndexFlatL2(dim)
    index_IVF = faiss.IndexIVFPQ(quantizer, dim, nlist, m, 8)
    index_IVF.train(data)                                    # index_IVF.is_trained=True， 建码本
    index_IVF.add(data)                                      # 添加索引
    index_IVF.nprobe = nprobe                                # 默认nprobe=1
    return index_IVF


def topN(self, vector, size):
    distance, idx = self.cpuIndex.search(vector, size)
    disList = distance.tolist()[0]
    idxList = idx.tolist()[0]
    result = {}
    for i, dis in zip(idxList, disList):
        # docId, simId, title, firstCategory, _, _, _, _, _, _ = self.idInfoMap[i]
        docId, simId, title, c, sc, mediaId, source, sourceName, expireTime, docType = self.idInfoMap[i]
        if simId not in result:
            # result[simId] = (docId, dis, title, firstCategory)
            result[simId] = (docId, dis, title, c, sc, mediaId, source, sourceName, expireTime, docType)

    return result


def merge_faiss_result(distance, indexId):
    print(distance.shape)
    dis_list = distance.reshape(1, -1)[0]
    print('dis_list.shape:',  len(dis_list))
    idx_list = indexId.reshape(1, -1)[0]
    result = {}
    for i, dis in zip(idx_list, dis_list):
        if i in result:
            result[i] = min(dis, result[i])
        else:
            result[i] = dis
    if len(idx_list) != len(result):
        print(idx_list)
    return result


if __name__ == "__main__":
    d = 32
    ni = 10000  # item num
    nu = 100  # user num
    k = 4
    np.random.seed(1234)
    item_embs = np.random.random((ni, d)).astype('float32')  # 使用随机数创建数据
    item_embs[:, 0] += np.arange(ni) / 1000.
    user_embs = np.random.random((nu, d)).astype('float32')  # 使用随机数创建查询数据
    user_embs[:, 0] += np.arange(nu) / 1000.
    n_v = 4  # mind模型后，每个user拥有的向量个数
    user_embs = user_embs.reshape(int(100 / n_v), n_v, -1)
    print('user_embs:', user_embs.shape)

    # 测试函数是否写对, 没有问题
    # a = ivf_flat(item_embs, d, k, 100)
    # a = index_flat(item_embs, d, k)
    # a = ivf_pq(item_embs, d, k, 100, 8)

    # mind的多个向量，分别求前k个，然后排序，需要把item信息（id, simid, mediaId等，存储到一个表里）
    nlist = 100  # 划分单元格
    nprobe = 10  # 搜索单元格
    m = 8
    faiss_index = ivf_PQ(item_embs, d, m, nlist, nprobe)
    i = 0
    for emb in user_embs[:4]:
        i += 1
        print('==== 第{}个user'.format(i))
        distance, idx = faiss_index.search(emb, k)  # emb: 4个向量。<class 'numpy.ndarray'> == (4, 32)
        merge_result = merge_faiss_result(distance, idx)
