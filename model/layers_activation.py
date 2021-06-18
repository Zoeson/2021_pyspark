#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Time : 2021/6/15 16:57
# Author : 
# File : layers_activation.py
# Software: PyCharm
# description:
"""
    DNN 要用到激活函数
    Author:    Weichen Shen,weichenswc@163.com
"""

import tensorflow  as tf
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras.initializers import Zeros

try:
    unicode
except NameError:
    unicode = str


# -----------------------------ctr: DIN-----------------------
class Dice(Layer):
    """The Data Adaptive Activation Function in DIN,which can be viewed as a generalization of PReLu and can adaptively adjust the rectified point according to distribution of input data.

      Input shape
        - Arbitrary. Use the keyword argument `input_shape` (tuple of integers, does not include the samples axis) when using this layer as the first layer in a model.

      Output shape
        - Same shape as the input.

      Arguments
        - **axis** : Integer, the axis that should be used to compute data distribution (typically the features axis).

        - **epsilon** : Small float added to variance to avoid dividing by zero.

      References
        - [Zhou G, Zhu X, Song C, et al. Deep interest network for click-through rate prediction[C]//Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. ACM, 2018: 1059-1068.](https://arxiv.org/pdf/1706.06978.pdf)
    """

    def __init__(self, axis=-1, epsilon=1e-9, **kwargs):
        self.axis = axis
        self.epsilon = epsilon
        super(Dice, self).__init__(**kwargs)

    def build(self, input_shape):
        self.bn = tf.keras.layers.BatchNormalization(
            axis=self.axis,
            epsilon=self.epsilon,
            center=False,
            scale=False)
        self.alphas = self.add_weight(shape=(input_shape[-1],),
                                      initializer=Zeros(),
                                      dtype=tf.float32,
                                      name='dice_alpha')  # name='alpha_'+self.name
        self.uses_learning_phase = True
        super(Dice, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, training=None, **kwargs):
        #  inputs batchNormal,  sigmoid. --> alpha*(1-sigmoid)*input+sigmoid*input
        inputs_normed = self.bn(inputs, training=training)
        x_p = tf.sigmoid(inputs_normed)
        return self.alphas * (1.0 - x_p) * inputs + x_p * inputs

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self, ):
        config = {'axis': self.axis, 'epsilon': self.epsilon}
        base_config = super(Dice, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def activation_layer(activation):
    if activation in ['dice', 'Dice']:
        act_layer = Dice()
    elif isinstance(activation, (str, unicode)):
        act_layer = tf.keras.layers.Activation(activation)
    elif issubclass(activation, Layer):
        act_layer = activation()
    else:
        raise ValueError("Invalid activation, found %s. You should use a str or a Activation class layer." % activation)
    return act_layer
