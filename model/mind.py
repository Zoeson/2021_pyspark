#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Time : 2021/6/11 20:49
# Author : 
# File : mind.py
# Software: PyCharm
# description:


from tensorflow.python.keras.layers import Concatenate
from tensorflow.python.keras.models import Model
from .feature_column import SparseFeat, DenseFeat, VarLenSparseFeat, build_input_features
from .inputs import create_embedding_matrix, embedding_lookup, varlen_embedding_lookup, get_varlen_pooling_list, \
    get_dense_input
from .core import PoolingLayer, CapsuleLayer, DNN, EmbeddingIndex, LabelAwareAttention, SampledSoftmaxLayer
from .layer_utils import NoMask, combined_dnn_input
import tensorflow as tf
from .utils import get_item_embedding


def tile_user_otherfeat(user_other_feature, k_max):
    return tf.tile(tf.expand_dims(user_other_feature, -2), [1, k_max, 1])


def MIND(user_feature_columns, item_feature_columns, num_sampled=5, k_max=2, p=1.0, dynamic_k=False,
         user_dnn_hidden_units=(64, 32), dnn_activation='relu', dnn_use_bn=False, l2_reg_dnn=0, l2_reg_embedding=1e-6,
         dnn_dropout=0, output_activation='linear', seed=1024):
    """

    :param user_feature_columns:
    :param item_feature_columns:
    :param num_sampled:      int, the number of classes to randomly sample per batch.
    :param k_max:            int, the max size of user interest embedding
    :param p:
    :param dynamic_k:        bool, whether or not use dynamic interest number
    :param user_dnn_hidden_units:
                             list,list of positive integer or empty list,
                             the layer number and units in each layer of user tower
    :param dnn_activation:   Activation function to use in deep net
    :param dnn_use_bn:       bool, BatchNormalization
    :param l2_reg_dnn:       L2 regularizer strength applied to DNN
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param dnn_dropout:
    :param output_activation: Activation function to use in output layer
    :param seed:
    :return:                  A Keras model instance.
    """

    if len(item_feature_columns) > 1:
        raise ValueError("Now MIND only support 1 item feature like item_id")

    # todo: ???????????? user_feature_columns
    # 1)??? feature_columns??????class???????????????
    sparse_feature_columns = list(
        filter(lambda x: isinstance(x, SparseFeat), user_feature_columns)) if user_feature_columns else []
    dense_feature_columns = list(
        filter(lambda x: isinstance(x, DenseFeat), user_feature_columns)) if user_feature_columns else []
    varlen_sparse_feature_columns = list(
        filter(lambda x: isinstance(x, VarLenSparseFeat), user_feature_columns)) if user_feature_columns else []

    # 2)SparseFeat --> Tensor()??? 3???feature_columns ==> 4???Tensor
    features = build_input_features(user_feature_columns)  # odict={'user':Input(), 'gender':Input(), ..}
    inputs_list = list(features.values())  # ??????Input:  [Input(), Input(),Input(),...]

    # todo: ????????????item???????????????????????????VarLenSparseFeat??????, ?????????
    #           ????????????item??????????????????????????????behavior_feature ==>  history_feature_columns
    #           ????????????item?????????            ?????????????????????Seq_emb?? ==> sparse_varlen_feature_columns
    # 1?????????item???????????????
    item_feature_column = item_feature_columns[0]  # ?????????????????????
    item_feature_name, item_vocabulary_size, item_embedding_dim = \
        item_feature_column.name, item_feature_column.vocabulary_size, item_feature_column.embedding_dim
    history_feature_list = [item_feature_name]  # ['item']
    history_fc_names = list(map(lambda x: "hist_" + x, history_feature_list))  # ['hist_item],
    # 2) ??????VarLenSparseFeat?????????????????? Profile_feature???behavior_feature
    history_feature_columns = []  # behavior_feature
    sparse_varlen_feature_columns = []  # ????????????
    for fc in varlen_sparse_feature_columns:
        feature_name = fc.name
        if feature_name in history_fc_names:
            history_feature_columns.append(fc)
        else:
            sparse_varlen_feature_columns.append(fc)
    seq_max_len = history_feature_columns[0].maxlen  # ???????????????item????????????????????????item????????????

    # todo: ?????? creat Embedding:?????????fc.Embedding_name??????key.item???user???hist_name?????????embedding_name?????????
    # SparseFeat+VarLenSparseFeat ???4??? ---> 3???Embedding
    #       example:Embedding:<tensorflow.python.keras.layers.embeddings.Embedding object at 0x00000279308F3FC8>
    embedding_matrix_dict = create_embedding_matrix(user_feature_columns + item_feature_columns, l2_reg_embedding,
                                                    seed=seed, prefix="")


    # todo: ??????embedding_lookup
    """
    embedding_matrix_dict :dict_keys(['user', 'gender', 'item'])
          Embedding.key: gender
          Embedding.name: sparse_emd_gender
          Embedding:<tensorflow.python.keras.layers.embeddings. Embedding object at 0x00000279308F1148>
    item_feature_columns = [SparseFeat('item', 3+1, embedding_dim=4)]
    item_features = Input(item_feature_columns)
    feature       = Input(user_feature_columns)
    history_feature_list = ['item']
    history_fc_names = ['hist_item']
    sparse_feature_columns = [SparseFeat(user, gender)]

                                     sparse_embedding_dict, sparse_input_dict, sparse_feature_columns, return_feat_list,
                                     return_feat_list(??????mask???feature???

    """
    # ## todo:??????????????????user????????????
    # ## todo:     1???SparseFeat
    # ## todo:     2) VarLenSparseFeat: ??????????????????
    # ## todo:     3) VarLenSparseFeat: ??????????????????
    # ## todo:     6)  DenseFeat
    # 1) user???SparseFeat???????????????????????????SparseFeat(user, gender)???
    dnn_input_emb_list = embedding_lookup(embedding_matrix_dict, features, sparse_feature_columns,
                                          mask_feat_list=history_feature_list, to_list=True)

    # 2)   user???VarLenSparseFeat?????????Behavior_feature
    keys_emb_list = embedding_lookup(embedding_matrix_dict, features, history_feature_columns, history_fc_names,
                                     history_fc_names, to_list=True)  # ['hist_item']

    #  3)  user???VarLenSparseFeat?????????????????????
    sequence_embed_dict = varlen_embedding_lookup(embedding_matrix_dict, features, sparse_varlen_feature_columns)
    """
    [embedding_dict[feature_name], features[fc.length_name], features[fc.weight_name]]) ??????
       key_input,                     key_length_input,              value_input = input_list
    """
    # ??? WeightedSequenceLayer ??????SequencePoolingLayer
    sequence_embed_list = get_varlen_pooling_list(sequence_embed_dict, features, sparse_varlen_feature_columns,
                                                  to_list=True)
    # 4) ??????????????? ???????????? ??????
    dnn_input_emb_list += sequence_embed_list

    # 6) user???DenseFeat????????? ??????
    dense_value_list = get_dense_input(features, dense_feature_columns)

    # todo: ??????CapsuleLayer???
    print('\n====================CapsuleLayer  ==============')
    history_emb = PoolingLayer()(NoMask()(keys_emb_list))  # ??????capsule?????????Pooling
    hist_len = features['hist_len']  # mask,
    high_capsule = CapsuleLayer(input_units=item_embedding_dim,
                                out_units=item_embedding_dim, max_len=seq_max_len,
                                k_max=k_max)((history_emb, hist_len))
    # todo: ????????????user_deep_input: = user(user+gender) + high_capsule(hist_item??????)
    if len(dnn_input_emb_list) > 0 or len(dense_value_list) > 0:
        # ?????????user????????? + high_capsule)
        #  high_capsule.shape([None,2, 4]) dnn_input_emb_list.shape([None, 1, 8]) 8=gender4+user4+ Dense0
        user_other_feature = combined_dnn_input(dnn_input_emb_list, dense_value_list)  # [None, 1,8]
        other_feature_tile = tf.keras.layers.Lambda(tile_user_otherfeat, arguments={'k_max': k_max})(
            user_other_feature)  # [None, 2, 8]

        # 2> ??????
        user_deep_input = Concatenate()([NoMask()(other_feature_tile), high_capsule])  # [None, 2, 12]
    else:
        user_deep_input = high_capsule

    # todo: ??????DNN:  user_embedding
    user_embeddings = DNN(user_dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn,
                          output_activation=output_activation, seed=seed, name='user_embedding')(user_deep_input)

    # todo: ??????item PoolingLayer
    # =======================================================
    # 1)  SparseFeat -->Tensor
    item_features = build_input_features(item_feature_columns)
    item_inputs_list = list(item_features.values())
    # 2) QUERY: LabelAwareAttention
    query_emb_list = embedding_lookup(embedding_matrix_dict, item_features, item_feature_columns, history_feature_list,
                                      history_feature_list, to_list=True)  # ['item']
    # 2) ??? item_features???embedding_lookup
    target_emb = PoolingLayer()(NoMask()(query_emb_list))
    # ==================================================================

    # 2). item_index
    #  todo: EmbeddingIndex:class?????????????????????EmbeddingIndex([0,1,2,3])(Tensor('item'))
    item_index = EmbeddingIndex(list(range(item_vocabulary_size)))(item_features[item_feature_name])

    # 3) item_embedding_weight
    item_embedding_matrix = embedding_matrix_dict[item_feature_name]
    item_embedding_weight = NoMask()(item_embedding_matrix(item_index))  # Embedding(idx)?????????embedding_lookup??????????????????
    pooling_item_embedding_weight = PoolingLayer()([item_embedding_weight])

    # todo: ?????? LabelAwareAttention ??? dynamic_k
    #     keys_emb_list: User???VarLenSparseFeat??????hist_name?????????
    #     target_emb = PoolingLayer()(NoMask()(query_emb_list)), query_emb_list = embedding_lookup()
    #           (item????????????item??????????????????)

    if dynamic_k:  # use dynamic interest number
        # input???tuple??????3???
        user_embedding_final = LabelAwareAttention(k_max=k_max, pow_p=p, )((user_embeddings, target_emb, hist_len))
    else:
        user_embedding_final = LabelAwareAttention(k_max=k_max, pow_p=p, )((user_embeddings, target_emb))

    # todo: ?????? output: SampledSoftmaxLayer
    output = SampledSoftmaxLayer(num_sampled=num_sampled)(
        [pooling_item_embedding_weight, user_embedding_final, item_features[item_feature_name]])
    #      weights                          input                    labels

    model = Model(inputs=inputs_list + item_inputs_list, outputs=output)
    # todo: RETURN:model
    model.__setattr__('user_input', inputs_list)
    model.__setattr__('user_embedding', user_embeddings)

    model.__setattr__('item_input', item_inputs_list)
    # x: tf.squeeze(tf.gather(item_embedding, x)
    model.__setattr__('item_embedding',
                      get_item_embedding(pooling_item_embedding_weight, item_features[item_feature_name]))
    return model