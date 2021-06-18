#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Time : 2021/6/11 20:49
# Author : 
# File : utils.py
# Software: PyCharm
# description:


import tensorflow as tf
from tensorflow.python.keras.layers import Lambda
from tensorflow.python.keras import backend as K


def get_item_embedding(item_embedding, item_input_layer):
    print("pooling_item_embedding_weight: {}\nitem_input_layer:{}".format(item_embedding, item_input_layer))
    b = Lambda(lambda x: tf.squeeze(tf.gather(item_embedding, x), axis=1))(
        item_input_layer)
    print('item_embedding:', b)
    return Lambda(lambda x: tf.squeeze(tf.gather(item_embedding, x), axis=1))(
        item_input_layer)


def sampledsoftmaxloss(y_true, y_pred):
    return K.mean(y_pred)


def recall_N(y_true, y_pred, N=50):
    return len(set(y_pred[:N]) & set(y_true)) * 1.0 / len(y_true)