#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Time : 2021/6/22 14:38
# Author : 
# File : test_pyecharts.py
# Software: PyCharm
# description:

import pyecharts.constants as constants
import pyecharts
print(pyecharts.__version__)
constants.CONFIGURATION['HOST'] = "https://cdn.kesci.com/nbextensions/echarts"