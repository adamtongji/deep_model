#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import os
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


def rel_error(x,y):
    # max 返回 array中最大值, maximum返回较大array或值
    return np.max(np.abs(x - y)/(np.maximum(1e-8, np.abs(x)+np.abs(y))))
# test
# a=np.arange(1,5); b=np.arange(1,9,2); rel_error(a,b)


input_size = 4
hidden_size = 10
num_classes = 3
num_inputs = 5


def first_one_layer_model():
    model={}
    # 给予附带初始值的权重矩阵
    model["w1"]=np.linspace(-0.2, 0.6, num=input_size*hidden_size)\
        .reshape(input_size, hidden_size)
    model['b1']=np.linspace(-0.3,0.7, num=hidden_size)
    model["w2"]=np.linspace(-0.4, 0.1, num=hidden_size*num_classes)\
        .reshape(hidden_size, num_classes)
    model["b2"]=np.linspace(-0.5,0.9, num=num_classes)
    return model


def first_data():
    x=np.linspace(-0.2,0.5, num=num_inputs*input_size)\
        .reshape(num_inputs, input_size)
    y=np.array([0,2,1,1,2])
    return x,y

x,y = first_data()
model = first_one_layer_model()
from nn.classifiers.neural_net import two_layer_net
from nn2.neural_net import two_layer_net

scores=two_layer_net(x, model,verbose=True)
print scores


