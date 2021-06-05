#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Time : 2021/6/5 15:08
# Author : 
# File : generate_hour_data.py
# Software: PyCharm
# description:
"""
之前想着是每一小时都把一周的数据准备好，写入hdfs,然后模型每小时直接去hdfs上读数据，
但是如果不能增量的进行数据读取，每次都往hdfs写入1周的数据，量太大，写不上
只能，每次写到hdfs一小时的数据，然后，down到本机内存，然后看内存的处理量

0605尝试一：
1. 写24个小时的数据到hdfs上，然后就够了
2. 写代码，尝试连续处理这24个小时的数据，包括去重，去掉user小于6次点击
MIND属于无序行为序列？没有回话的概念

"""