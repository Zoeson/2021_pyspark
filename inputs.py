#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Time : 2021/6/11 20:46
# Author : 
# File : inputs.py
# Software: PyCharm
# description:

from .feature_column import SparseFeat, VarLenSparseFeat, DenseFeat
from tensorflow.python.keras.layers import Embedding, Lambda
from tensorflow.python.keras.regularizers import l2
from collections import defaultdict
from deepmatch.layers.utils import Hash
from itertools import chain
from .layers.sequence import SequencePoolingLayer, WeightedSequenceLayer


def create_embedding_matrix(feature_columns, l2_reg, seed, prefix="", seq_mask_zero=True):
    """
        1. 根据 feature_columns的class的不同，进行分类
        2. creat_embedding_dict，按照SpareFeat和VarLenSparseFeat生成_emb_或者_seq_emb类型的Embedding,
    :param feature_columns:
    :param l2_reg:
    :param seed:
    :param prefix:
    :param seq_mask_zero:
    :return:
        _emb_或者_seq_emb类型的Embedding
    """
    sparse_feature_columns = list(
        filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if feature_columns else []
    varlen_sparse_feature_columns = list(
        filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns)) if feature_columns else []
    sparse_emb_dict = create_embedding_dict(sparse_feature_columns, varlen_sparse_feature_columns, seed,
                                            l2_reg, prefix=prefix + 'sparse', seq_mask_zero=seq_mask_zero)
    return sparse_emb_dict


def create_embedding_dict(sparse_feature_columns, varlen_sparse_feature_columns, seed, l2_reg,
                          prefix='sparse_', seq_mask_zero=True):
    """
        1. sparseFeat：      --> sparse_emd_     + name的Embedding
        2. varLenSparseFeat: --> sparse_seq_emb_ + name 的Embedding
        注意：feat.embedding_name
    :param sparse_feature_columns:
    :param varlen_sparse_feature_columns:
    :param seed:
    :param l2_reg:
    :param prefix:
    :param seq_mask_zero:
    :return:
    """
    sparse_embedding = {}
    print('\n============================\n-----create_embedding_dict-------')
    for fc in sparse_feature_columns:
        emb = Embedding(fc.vocabulary_size, fc.embedding_dim,
                        embeddings_initializer=fc.embeddings_initializer,
                        embeddings_regularizer=l2(l2_reg),
                        name=prefix + '_emd_' + fc.embedding_name)
        emb.trainable = fc.trainable
        sparse_embedding[fc.embedding_name] = emb
    print('sparse_feature_columns的 embbeding的个数：{}  \n---------keys():{}，  \n---------{}'.format(len(sparse_embedding),
                                                                               sparse_embedding.keys(),
                                                                               sparse_embedding))
    if varlen_sparse_feature_columns and len(varlen_sparse_feature_columns) > 0:
        for fc in varlen_sparse_feature_columns:
            emb = Embedding(fc.vocabulary_size, fc.embedding_dim,
                            embeddings_initializer=fc.embeddings_initializer,
                            embeddings_regularizer=l2(l2_reg),
                            name=prefix + '_seq_emb_' + fc.name,
                            mask_zero=seq_mask_zero)
            emb.trainable = fc.trainable
            sparse_embedding[fc.embedding_name] = emb
    print('varlen_sparse_feature_columns之后的 embbeding的个数：{}  \n---------keys():{} \n---------{}'.format(len(sparse_embedding),
                                                                                         sparse_embedding.keys(),
                                                                                         sparse_embedding))

    return sparse_embedding


def embedding_lookup(sparse_embedding_dict, sparse_input_dict, sparse_feature_columns, return_feat_list=(),
                     mask_feat_list=(), to_list=False):
    """
        参数3：SparseFeat/VarLenSparseFeat:
            for fc in sparse_feature_columns: (user/item)
        参数2：Input() :
            lookup_idx = Hash(fc.vocabulary_size, mask_zero=())(sparse_input_dict[fc.name])
        参数1：最后return:
            sparse_embedding_dict[fc.embedding_name](lookup_idx)
    :param sparse_embedding_dict:      embedding_matrix_dict:Embedding
    :param sparse_input_dict:          user/item: Input()
    :param sparse_feature_columns:     选的是这里面的特征！！！！！！SpareseFeat或者VarLenSparseFeat
    :param return_feat_list:
        如果=（），返回sparse_feature_columns所有特征， else:只能返回list的特征（todo history_feature_list??? ）
    :param mask_feat_list:             需要做mask的特征？？？history_feature_list
    :param to_list:
    :return:
    """

    group_embedding_dict = defaultdict(list)
    print('\n=================embedding_lookup=====')
    for fc in sparse_feature_columns:  # item
        print("fc.embedding_name: {}  fc:{}".format(fc.embedding_name, fc))
        feature_name = fc.name
        embedding_name = fc.embedding_name
        if len(return_feat_list) == 0 or feature_name in return_feat_list:
            if fc.use_hash:
                print('=== embedding_lookup:\n    feature_name:{}   \n    sparse_input_dict[feature_name]:{}'
                      .format(feature_name, sparse_input_dict[feature_name]))
                # todo: mask_feat_list？
                lookup_idx = Hash(fc.vocabulary_size, mask_zero=(feature_name in mask_feat_list))(
                    sparse_input_dict[feature_name])
            else:  # 默认fc.use_hash = False
                print("sparse.use_hash = False")
                lookup_idx = sparse_input_dict[feature_name]

            print('lookup_idx: type:{},   {}'.format(type(lookup_idx), lookup_idx))
            print("sparses_embedding_dict[embedding_name]:{}".format(sparse_embedding_dict[embedding_name]))
            print('embedding_dict[](look_idx):{}'.format(sparse_embedding_dict[embedding_name](lookup_idx)))
            group_embedding_dict[fc.group_name].append(sparse_embedding_dict[embedding_name](lookup_idx))

    print("group_embedding_dict: {}".format(group_embedding_dict))
    if to_list:
        return list(chain.from_iterable(group_embedding_dict.values()))
    return group_embedding_dict


def varlen_embedding_lookup(embedding_dict, sequence_input_dict, varlen_sparse_feature_columns):
    varlen_embedding_vec_dict = {}
    for fc in varlen_sparse_feature_columns:
        feature_name = fc.name
        embedding_name = fc.embedding_name
        if fc.use_hash:
            lookup_idx = Hash(fc.vocabulary_size, mask_zero=True)(sequence_input_dict[feature_name])
        else:
            lookup_idx = sequence_input_dict[feature_name]
        varlen_embedding_vec_dict[feature_name] = embedding_dict[embedding_name](lookup_idx)
    return varlen_embedding_vec_dict


def get_varlen_pooling_list(embedding_dict, features, varlen_sparse_feature_columns, to_list=False):
    """
    只要
    :param embedding_dict:
    :param features:
    :param varlen_sparse_feature_columns:
    :param to_list:
    :return:
    """
    print('   ========varlen_pooling_list ==========')
    pooling_vec_list = defaultdict(list)
    for fc in varlen_sparse_feature_columns:
        feature_name = fc.name
        combiner = fc.combiner
        feature_length_name = fc.length_name
        if feature_length_name is not None:
            if fc.weight_name is not None:
                print('        ----length_name=any_len, wight_name=any_norm ---？？      ')
                seq_input = WeightedSequenceLayer(weight_normalization=fc.weight_norm)(
                    [embedding_dict[feature_name], features[feature_length_name], features[fc.weight_name]])
            else:
                seq_input = embedding_dict[feature_name]
            vec = SequencePoolingLayer(combiner, supports_masking=False)(
                [seq_input, features[feature_length_name]])
        else:  # todo: fc.length_name  is None, --》 supports_masking=True，--》要开始用mask
            # todo: 问题来了 mask在哪儿
            if fc.weight_name is not None:
                seq_input = WeightedSequenceLayer(weight_normalization=fc.weight_norm, supports_masking=True)(
                    [embedding_dict[feature_name], features[fc.weight_name]])
            else:
                seq_input = embedding_dict[feature_name]
            vec = SequencePoolingLayer(combiner, supports_masking=True)(
                seq_input)
        pooling_vec_list[fc.group_name].append(vec)
    if to_list:
        return chain.from_iterable(pooling_vec_list.values())
    return pooling_vec_list


def get_dense_input(features, feature_columns):
    dense_feature_columns = list(
        filter(lambda x: isinstance(x, DenseFeat), feature_columns)) if feature_columns else []
    dense_input_list = []
    for fc in feature_columns:
        if fc.transform_fn is None:
            dense_input_list.append(features[fc.name])
        else:
            transform_result = Lambda(fc.transform_fn)(features[fc.name])
            dense_input_list.append(transform_result)

    return dense_input_list
