#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Time : 2021/6/15 16:56
# Author : 
# File : layer_utils.py
# Software: PyCharm
# description:
from tensorflow.python.keras.layers import Flatten
import tensorflow as tf


class Hash(tf.keras.layers.Layer):
    """
    mask_zero --> num_buckets --> hash_x
    """

    def __init__(self, num_buckets, mask_zero=False, **kwargs):
        print('\n ======== SparseFeat  use hash:  _init__====')
        self.num_buckets = num_buckets
        self.mask_zero = mask_zero
        super(Hash, self).__init__(**kwargs)

    def build(self, input_shape):  # todo:???不懂
        # Be sure to call this somewhere!  todo:???
        super(Hash, self).build(input_shape)

    def call(self, x, mask=None, **kwargs):
        # todo: tf.string: tf.string
        #   tf.strings =<module 'tensorflow_core._api.v2.strings' from
        #   'C:\\Users\\heling\\AppData\\Local\\Continuum\\anaconda3\\envs\\tf2\\lib\\site-packages\\tensorflow_core\\_api\\v2\\strings\\__init__.py'>
        if x.dtype != tf.string:
            zero = tf.as_string(tf.zeros([1], dtype=x.dtype))
            # type(zero) = <class 'tensorflow.python.framework.ops.EagerTensor'>
            x = tf.as_string(x, )
        else:
            zero = tf.as_string(tf.zeros([1], dtype='int32'))

        # todo: num_buckets:???
        num_buckets = self.num_buckets if not self.mask_zero else self.num_buckets - 1
        try:
            hash_x = tf.string_to_hash_bucket_fast(x, num_buckets, name=None)  # weak hash
        except:
            hash_x = tf.strings.to_hash_bucket_fast(x, num_buckets,
                                                    name=None)  # weak hash
        if self.mask_zero:
            mask = tf.cast(tf.not_equal(x, zero), dtype='int64')
            hash_x = (hash_x + 1) * mask

        return hash_x

    def get_config(self, ):
        config = {'num_buckets': self.num_buckets, 'mask_zero': self.mask_zero, }
        base_config = super(Hash, self).get_config()
        print("============Hash:====\nconfig:{}\nbase_config:{}".format(config, base_config))
        return dict(list(base_config.items()) + list(config.items()))


class NoMask(tf.keras.layers.Layer):
    """
    todo: ？？？
    """
    def __init__(self, **kwargs):
        super(NoMask, self).__init__(**kwargs)

    def build(self, input_shape):
        super(NoMask, self).build(input_shape)

    def call(self, x, mask=None, **kwargs):
        return x

    def comput_mask(self, inputs, mask):
        return None


def reduce_mean(input_tensor,
                axis=None,
                keep_dims=False,
                name=None,
                reduction_indices=None):
    try:
        return tf.reduce_mean(input_tensor,
                              axis=axis,
                              keep_dims=keep_dims,
                              name=name,
                              reduction_indices=reduction_indices)
    except TypeError:
        return tf.reduce_mean(input_tensor,
                              axis=axis,
                              keepdims=keep_dims,
                              name=name)


def reduce_sum(input_tensor,
               axis=None,
               keep_dims=False,
               name=None,
               reduction_indices=None):
    try:
        return tf.reduce_sum(input_tensor,
                             axis=axis,
                             keep_dims=keep_dims,
                             name=name,
                             reduction_indices=reduction_indices)
    except TypeError:
        return tf.reduce_sum(input_tensor,
                             axis=axis,
                             keepdims=keep_dims,
                             name=name)


def reduce_max(input_tensor,
               axis=None,
               keep_dims=False,
               name=None,
               reduction_indices=None):
    try:
        return tf.reduce_max(input_tensor,
                             axis=axis,
                             keep_dims=keep_dims,
                             name=name,
                             reduction_indices=reduction_indices)
    except TypeError:
        return tf.reduce_max(input_tensor,
                             axis=axis,
                             keepdims=keep_dims,
                             name=name)


def div(x, y, name=None):
    try:
        return tf.div(x, y, name=name)
    except AttributeError:
        return tf.divide(x, y, name=name)


def softmax(logits, dim=-1, name=None):
    try:
        return tf.nn.softmax(logits, dim=dim, name=name)
    except TypeError:
        return tf.nn.softmax(logits, axis=dim, name=name)


def concat_func(inputs, axis=-1, mask=False):
    """
    todo:  ???
    :param inputs:
    :param axis:
    :param mask:
    :return:
    """
    if not mask:
        inputs = list(map(NoMask(), inputs))
    if len(inputs) == 1:
        return inputs[0]
    else:
        return tf.keras.layers.Concatenate(axis=axis)(inputs)


# --------------------------ctr---------------------
def combined_dnn_input(sparse_embedding_list, dense_value_list):
    """
    拼接 特征，要flatten和concatenate
    user的除VarLenSparseFeat中满足hist_name（hist_item）以外的所有featurs
    :param sparse_embedding_list:  SparseFeat+VarlenSparseFeat不满足hist_name的SparseFeat
    :param dense_value_list:       DenseFeat
    :return:
    """
    if len(sparse_embedding_list) > 0 and len(dense_value_list) > 0:
        sparse_dnn_input = Flatten()(concat_func(sparse_embedding_list))
        dense_dnn_input = Flatten()(concat_func(dense_value_list))
        return concat_func([sparse_dnn_input, dense_dnn_input])
    elif len(sparse_embedding_list) > 0:
        return Flatten()(concat_func(sparse_embedding_list))
    elif len(dense_value_list) > 0:
        return Flatten()(concat_func(dense_value_list))
    else:
        raise NotImplementedError("dnn_feature_columns can not be empty list!!")