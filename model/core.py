#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Time : 2021/6/15 16:50
# Author : 
# File : core.py
# Software: PyCharm
# description:
"""
layers
"""

from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras.initializers import RandomNormal, glorot_normal, Zeros
from tensorflow.python.keras.regularizers import l2
import tensorflow as tf
from .layer_utils import concat_func, reduce_max, reduce_sum, reduce_mean, softmax, div

from .layers_activation import activation_layer


# ----------------------recall-----------------
class PoolingLayer(Layer):
    def __init__(self, mode='mean', supports_masking=False, **kwargs):
        if mode not in ['mean', 'max', 'sum']:
            raise ValueError("mode must be mean max or sum")
        self.mode = mode
        self.eps = tf.constant(1e-8, tf.float32)
        super(PoolingLayer, self).__init__(**kwargs)

        self.supports_masking = supports_masking

    def build(self, input_shape):
        super(PoolingLayer, self).build(input_shape)

    def call(self, seq_value_len_list, mask=None, **kwargs):
        print('    -----------CALL: PoolingLayer ')
        if not isinstance(seq_value_len_list, list):
            seq_value_len_list = [seq_value_len_list]
        if len(seq_value_len_list) == 1:
            return seq_value_len_list[0]
        expand_seq_value_len_list = list(map(lambda x: tf.expand_dims(x, axis=-1), seq_value_len_list))
        a = concat_func(expand_seq_value_len_list)
        if self.mode == 'mean':
            hist = reduce_mean(a, axis=-1, )
        if self.mode == 'max':
            hist = reduce_max(a, axis=-1, )
        if self.mode == 'sum':
            hist = reduce_sum(a, axis=-1, )
        return hist

    def get_config(self):
        config = {'mode': self.mode, 'supports_masking': self.supports_masking}
        base_config = super(PoolingLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class CapsuleLayer(Layer):
    """
            b(i,j) = self.routing_logits
            S(i,j)=self.bilinear_mapping_matrix
            C(low_capsule) = behavior_embedding,
            C(high_capsul) = interest_capsules=squeeze(Z)
    """
    def __init__(self, input_units, out_units, max_len, k_max, iteration_times=3, init_std=1.0, **kwargs):
        self.input_units = input_units
        self.out_units = out_units
        self.max_len = max_len
        self.k_max = k_max
        self.iteration_times = iteration_times
        self.init_std = init_std
        super(CapsuleLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # MIND函数注释：k_max: int, the max size of user interest embedding
        self.routing_logits = self.add_weight(shape=[1, self.k_max, self.max_len],
                                              initializer=RandomNormal(stddev=self.init_std),
                                              trainable=False,
                                              name="B",
                                              dtype=tf.float32)
        self.bilinear_mapping_matrix = self.add_weight(shape=[self.input_units, self.out_units],
                                                       initializer=RandomNormal(stddev=self.init_std),
                                                       name="S", dtype=tf.float32)
        super(CapsuleLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """
        :param inputs:
         history_emb: low_capsule:
            hist_len: mask
        :param kwargs:
        :return:
        """
        print('  ------ call------')
        print('  --  inputs:    {}'.format(inputs))
        behavior_embedding, seq_len = inputs
        batch_size = tf.shape(behavior_embedding)[0]
        seq_len_tile = tf.tile(seq_len, [1, self.k_max])

        for i in range(self.iteration_times):
            # 1. b_ij = c_h * S* c_l
            # 2. w = softmax(b_ij)
            # 3. Z= w * S*c_l
            # 4. c_h=squeeze(Z)
            mask = tf.sequence_mask(seq_len_tile, self.max_len)
            pad = tf.ones_like(mask, dtype=tf.float32) * (-2 ** 32 + 1)

            # bij = C(high_capsule) * S(i,j) * C(low_capsule)
            routing_logits_with_padding = tf.where(mask, tf.tile(self.routing_logits, [batch_size, 1, 1]), pad)
            # 2. w(i,j) = softmax(bij)
            weight = tf.nn.softmax(routing_logits_with_padding)

            # 3  Z=W*S*C(low_capsule)
            behavior_embedding_mapping = tf.tensordot(behavior_embedding, self.bilinear_mapping_matrix, axes=1)
            Z = tf.matmul(weight, behavior_embedding_mapping)

            # 4. squeeze(Z)
            interest_capsules = squash(Z)

            # 1. b_ij = C(high_capsule) * S(i,j) * C(low_capsule)
            #         = c_h  * behavior_embedding_mapping = c_h * (c_l * S)
            delta_routing_logits = reduce_sum(
                tf.matmul(interest_capsules, tf.transpose(behavior_embedding_mapping, perm=[0, 2, 1])),
                axis=0, keep_dims=True)
            self.routing_logits.assign_add(delta_routing_logits)

        interest_capsules = tf.reshape(interest_capsules, [-1, self.k_max, self.out_units])
        return interest_capsules

    def compute_output_shape(self, input_shape):
        return None, self.k_max, self.out_units

    def get_config(self, ):
        config = {'input_units': self.input_units, 'out_units': self.out_units, 'max_len': self.max_len,
                  'k_max': self.k_max, 'iteration_times': self.iteration_times, "init_std": self.init_std}
        base_config = super(CapsuleLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def squash(inputs):
    vec_squared_norm = reduce_sum(tf.square(inputs), axis=-1, keep_dims=True)
    scalar_factor = vec_squared_norm / (1 + vec_squared_norm) / tf.sqrt(vec_squared_norm + 1e-8)
    vec_squashed = scalar_factor * inputs
    return vec_squashed


class LabelAwareAttention(Layer):
    def __init__(self, k_max, pow_p=1, **kwargs):
        self.k_max = k_max
        self.pow_p = pow_p
        super(LabelAwareAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.embedding_size = input_shape[0][-1]
        super(LabelAwareAttention, self).build(input_shape)

    def call(self, inputs, training=None, **kwargs):
        keys = inputs[0]
        query = inputs[1]  # query干啥的
        weight = reduce_sum(keys * query, axis=-1, keep_dims=True)
        weight = tf.pow(weight, self.pow_p)
        input_k = tf.math.log1p(tf.cast(inputs[2], dtype='float32')) / tf.math.log(2.)
        if len(inputs) == 3:
            k_user = tf.cast(tf.maximum(1.,
                                        tf.minimum(tf.cast(self.k_max, dtype='float32'),
                                                   tf.math.log1p(tf.cast(inputs[2], dtype='float32')) / tf.math.log(2.)
                                                   )
                                        ),
                             dtype='int64')
            seq_mask = tf.transpose(tf.sequence_mask(k_user, self.k_max), [0, 2, 1])
            """
            tf.sequence_mask([1, 3, 2], 5)  
                                [[True, False, False, False, False],  1
                                 [True, True, True, False, False],     3
                                 [True, True, False, False, False]]    2
            按[0.2.1]的排序进行倒置
            """
            padding = tf.ones_like(seq_mask, dtype=tf.float32) * (-2 ** 32 + 1)
            weight = tf.where(seq_mask, weight, padding)

        weight = softmax(weight, dim=1, name='weight')
        output = reduce_sum(keys * weight, axis=1)
        return output

    def compute_output_shape(self, input_shape):
        return None, self.embedding_size

    def get_config(self):
        config = {'k_max': self.k_max, 'pow_p': self.pow_p}
        base_config = super(LabelAwareAttention, self).get_config()
        return dict(list(config.items()) + list(base_config.items()))


class SampledSoftmaxLayer(Layer):
    def __init__(self, num_sampled=5, **kwargs):
        self.num_sampled = num_sampled
        super(SampledSoftmaxLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.size = input_shape[0][0]
        self.zero_bias = self.add_weight(shape=[self.size],
                                         initializer=Zeros,
                                         dtype=tf.float32,
                                         trainable=False,
                                         name='bias')
        super(SampledSoftmaxLayer, self).build(input_shape)

    def call(self, inputs_with_label_idx, training=None, **kwargs):
        embeddings, inputs, label_idx = inputs_with_label_idx
        loss = tf.nn.sampled_softmax_loss(weights=embeddings,
                                          biases=self.zero_bias,
                                          labels=label_idx,
                                          inputs=inputs,
                                          num_sampled=self.num_sampled,
                                          num_classes=self.size)
        return tf.expand_dims(loss, axis=1)

    def compute_output_shape(self, input_shape):
        return None, 1

    def get_config(self):
        config = {'num_sampled':self.num_sampled}
        base_config = super(SampledSoftmaxLayer, self).get_config()
        return dict(list(config.items()) + list(base_config.items()))


# todo: call的参数：x在哪里呢？？
class EmbeddingIndex(Layer):
    def __init__(self, index, **kwargs):
        self.index = index
        super(EmbeddingIndex, self).__init__(**kwargs)

    def build(self, input_shape):
        super(EmbeddingIndex, self).build(input_shape)

    def call(self, x, **kwargs):
        return tf.constant(self.index)

    def get_config(self, ):
        config = {'index': self.index, }
        base_config = super(EmbeddingIndex, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# ----------------------ctr------------------------
# DNN需要分清各个维度，维度，维度！！！
class DNN(Layer):
    def __init__(self, hidden_units, activation='relu', l2_reg=0, dropout_rate=0, use_bn=False, output_activation=None,
                 seed=1024, **kwargs):
        self.hidden_units = hidden_units
        self.activation = activation
        self.l2_reg = l2_reg
        self.dropout_rate = dropout_rate
        self.use_bn = use_bn
        self.output_activation = output_activation
        self.seed = seed

        super(DNN, self).__init__(**kwargs)

    def build(self, input_shape):
        """
        生成每个隐层（包含输入层在内）的： kernals, bias， bn_layers, dropout_layers, activation_layers,
        :param input_shape:
        :return:
        """
        print('------------------DNN call----------------------')
        input_size = input_shape[-1]

        hidden_units = [int(input_size)] + list(self.hidden_units)  # 输入层 算是第一隐层

        self.kernels = [self.add_weight(name='kernel' + str(i),
                                        shape=(hidden_units[i], hidden_units[i + 1]),
                                        initializer=glorot_normal(seed=self.seed),
                                        regularizer=l2(self.l2_reg),
                                        trainable=True) for i in range(len(self.hidden_units))]

        self.bias = [self.add_weight(name='bias' + str(i),
                                     shape=(self.hidden_units[i],),
                                     initializer=Zeros(),
                                     trainable=True) for i in range(len(self.hidden_units))]

        if self.use_bn:
            self.bn_layers = [tf.keras.layers.BatchNormalization() for _ in range(len(self.hidden_units))]

        self.dropout_layers = [tf.keras.layers.Dropout(self.dropout_rate, seed=self.seed + i) for i in
                               range(len(self.hidden_units))]

        self.activation_layers = [activation_layer(self.activation) for _ in range(len(self.hidden_units))]
        if self.output_activation:
            self.activation_layers[-1] = activation_layer(self.output_activation)

        super(DNN, self).build(input_shape)

    def call(self, inputs, training=None, **kwargs):
        # 开始计算DNN
        deep_input = inputs

        for i in range(len(self.hidden_units)):
            # 输入*kernel+bias 过bn_layer, 过activation, 过dropout_layer ==>输出=下一次的输入
            fc = tf.nn.bias_add(tf.tensordot(deep_input, self.kernels[i], axes=(-1, 0)),
                                self.bias[i])
            if self.use_bn:
                fc = self.bn_layers[i](fc, training=training)

            fc = self.activation_layers[i](fc)
            fc = self.dropout_layers[i](fc, training=training)

            deep_input = fc
        return deep_input

    def compute_output_shape(self, input_shape):
        if len(self.hidden_units) > 0:
            shape = input_shape[-1] + (self.hidden_units[-1],)
        else:
            shape = input_shape
        return tuple(shape)

    def get_config(self, ):
        config = {'activation': self.activation, 'hidden_units': self.hidden_units,
                  'l2_reg': self.l2_reg, 'use_bn': self.use_bn, 'dropout_rate': self.dropout_rate,
                  'output_activation': self.output_activation, 'seed': self.seed}
        base_config = super(DNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SequencePoolingLayer(Layer):
    """The SequencePoolingLayer is used to apply pooling operation(sum,mean,max) on variable-length sequence feature/multi-value feature.

      Input shape
        - A list of two  tensor [seq_value,seq_len]

        - seq_value is a 3D tensor with shape: ``(batch_size, T, embedding_size)``

        - seq_len is a 2D tensor with shape : ``(batch_size, 1)``,indicate valid length of each sequence.

      Output shape
        - 3D tensor with shape: ``(batch_size, 1, embedding_size)``.

      Arguments
        - **mode**:str.Pooling operation to be used,can be sum,mean or max.

        - **supports_masking**:If True,the input need to support masking.
    """

    def __init__(self, mode='mean', supports_masking=False, **kwargs):

        if mode not in ['sum', 'mean', 'max']:
            raise ValueError("mode must be sum or mean")
        self.mode = mode
        self.eps = tf.constant(1e-8, tf.float32)
        super(SequencePoolingLayer, self).__init__(**kwargs)

        self.supports_masking = supports_masking

    def build(self, input_shape):
        if not self.supports_masking:
            self.seq_len_max = int(input_shape[0][1])
        super(SequencePoolingLayer, self).build(
            input_shape)  # Be sure to call this somewhere!

    def call(self, seq_value_len_list, mask=None, **kwargs):
        if self.supports_masking:
            if mask is None:
                raise ValueError(
                    "When supports_masking=True,input must support masking")
            uiseq_embed_list = seq_value_len_list
            mask = tf.cast(mask, tf.float32)  # tf.to_float(mask)
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
            return (None, 1, input_shape[-1])
        else:
            return (None, 1, input_shape[0][-1])

    def compute_mask(self, inputs, mask):
        return None

    def get_config(self, ):
        config = {'mode': self.mode, 'supports_masking': self.supports_masking}
        base_config = super(SequencePoolingLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))