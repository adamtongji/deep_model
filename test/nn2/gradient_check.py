#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from random import randrange

def eval_numerical_gradient(f, x, verbose=True, h=0.00001):
    # x 是需要重新计算梯度的点
    # f是接受一个参数的函数
    # h 梯度
    fx = f(x)
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index
