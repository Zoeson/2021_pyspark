#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Time : 2021/6/15 17:08
# Author : 
# File : sequence.py
# Software: PyCharm
# description:

from tensorflow.python.keras.layers import Layer
import tensorflow as tf
from .layer_utils import reduce_mean, reduce_sum, reduce_max, div, softmax


class SequencePoolingLayer(Layer):

    def __init__(self, mode='mean', supports_masking=False, **kwargs):
        if mode not in ['mean', 'sum', 'max']:
            raise ValueError("mode must be sum or mean or max")
        self.mode = mode
        self.eps = tf.constant(1e-8, tf.float32)  # <tf.Tensor: shape=(), dtype=float32, numpy=1e-08>
        super(SequencePoolingLayer, self).__init__(**kwargs)

        self.supports_masking = supports_masking

    def build(self, input_shape):
        if not self.supports_masking:
            self.seq_len_max = int(input_shape[0][1])
        # todo: Be sure to call this somewhere!???
        #  WHY?
        #  HOW?
        super(SequencePoolingLayer, self).build(input_shape)

    def call(self, seq_value_len_list, mask=None, **kwargs):
        if self.supports_masking:
            if mask is None:
                raise ValueError("When supports_masking=True,input must support masking")
            uiseq_embed_list = seq_value_len_list
            mask = tf.cast(mask, tf.float32)
            user_behavior_length = reduce_sum(mask, axis=-1, keep_dims=True)

            mask = tf.expand_dims(mask, axis=2)
        else:
            uiseq_embed_list, user_behavior_length = seq_value_len_list

            mask = tf.sequence_mask(user_behavior_length,
                                    self.seq_len_max, dtype=tf.float32)
            mask = tf.transpose(mask, (0, 2, 1))

        embedding_size = uiseq_embed_list.shape[-1]
        mask = tf.tile(mask, [1, 1, embedding_size])

        if self.mode == "max":
            hist = uiseq_embed_list - (1 - mask) * 1e9
            return reduce_max(hist, 1, keep_dims=True)

        hist = reduce_sum(uiseq_embed_list * mask, 1, keep_dims=False)

        if self.mode == "mean":
            hist = div(hist, tf.cast(user_behavior_length, tf.float32) + self.eps)

        hist = tf.expand_dims(hist, axis=1)
        return hist

    def compute_output_shape(self, input_shape):
        if self.supports_masking:
            return None, 1, input_shape[-1]
        else:
            return None, 1, input_shape[0][-1]

    def compute_mask(self, inputs, mask):
        return None

    def get_config(self, ):
        config = {'mode': self.mode, 'supports_masking': self.supports_masking}
        base_config = super(SequencePoolingLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class WeightedSequenceLayer(Layer):
    def __init__(self, weight_normalization=True, supports_masking=False, **kwargs):
        print("   ------ 进入 WeightedSequenceLayer  的 __init__ -----")
        super(WeightedSequenceLayer, self).__init__(**kwargs)
        self.weight_normalization = weight_normalization
        self.supports_masking = supports_masking

    def build(self, input_shape):
        print("   ------ 进入 WeightedSequenceLayer  的 build -----")
        if not self.supports_masking:
            print("     ---input_shape: {}".format(input_shape))
            self.seq_len_max = int(input_shape[0][1])
        # todo: Be sure to call this somewhere!
        super(WeightedSequenceLayer, self).build(input_shape)

    def call(self, input_list, mask=None, **kwargs):
        """
        1. supports_masking: 必须有mask:  input_list=2维？
                       else:  不支持mask: input_list=3维.
                                        key_length_input?? self.seq_len_max??构建mask?
        2. weight_normalization: -->padding: tf.ones_like or tf.zeros_like
                            mask, value_input, padding  --> 更新 value_input
        :param input_list:
        :param mask:
        :param kwargs:
        :return:
        """
        print("  ---- 进入 WeightedSequenceLayer  的call------")
        if self.supports_masking:
            if mask is None:
                raise ValueError("When supports_masking=True,input must support masking")
            key_input, value_input = input_list
            mask = tf.expand_dims(mask[0], axis=2)
        else:
            # todo: [embedding_dict[feature_name], features[fc.length_name], features[fc.weight_name]]
            key_input, key_length_input, value_input = input_list
            mask = tf.sequence_mask(key_length_input, self.seq_len_max, dtype=tf.bool)
            mask = tf.transpose(mask, (0, 2, 1))

        embedding_size = key_input.shape[-1]

        if self.weight_normalization:
            paddings = tf.ones_like(value_input) * (-2 ** 32 + 1)
        else:
            paddings = tf.zeros_like(value_input)
        value_input = tf.where(mask, value_input, paddings)
        print('\n ====mask, value_input, paddings----\nmask: {}\nvalue_input:  {}\nkey_input: {}\npaddings: {}'
              .format(mask, value_input, key_input, paddings))

        if self.weight_normalization:
            value_input = softmax(value_input, dim=1)

        print("value_input.shape: {}".format(value_input.shape))
        if len(value_input.shape) == 2:
            value_input = tf.expand_dims(value_input, axis=2)
            value_input = tf.tile(value_input, [1, 1, embedding_size])

        return tf.multiply(key_input, value_input)

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def compute_mask(self, inputs, mask):
        if self.supports_masking:
            return mask[0]
        else:
            return None

    def get_config(self, ):
        # todo: 看不懂
        config = {'weight_normalization': self.weight_normalization, 'supports_masking': self.supports_masking}
        base_config = super(WeightedSequenceLayer, self).get_config()
        print("GET CONFIG->BASE CONFIG:{}".format(base_config))
        return dict(list(base_config.items()) + list(config.items()))