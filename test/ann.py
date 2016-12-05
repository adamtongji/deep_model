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
# from nn.classifiers.neural_net import two_layer_net
from nn2.neural_net import two_layer_net

scores=two_layer_net(x, model,verbose=True)
print scores


correct_scores = [[-0.5328368, 0.20031504, 0.93346689],
 [-0.59412164, 0.15498488, 0.9040914 ],
 [-0.67658362, 0.08978957, 0.85616275],
 [-0.77092643, 0.01339997, 0.79772637],
 [-0.89110401, -0.08754544, 0.71601312]]

# 我们前向运算计算得到的得分和实际的得分应该差别很小才对
print '前向运算得到的得分和实际的得分差别:'
print np.sum(np.abs(scores - correct_scores))

reg = 0.1
loss, _ = two_layer_net(x, model, y, reg)
correct_loss = 1.38191946092

# 应该差值是很小的
print '我们计算到的损失和真实的损失值之间差别:'
print np.sum(np.abs(loss - correct_loss))

from nn.gradient_check import eval_numerical_gradient

loss, grads = two_layer_net(x, model, y, reg)

# 各参数应该比 1e-8 要小才保险
for param_name in grads:
    param_grad_num = eval_numerical_gradient(lambda W: two_layer_net(x, model, y, reg)[0], model[param_name],
                                             verbose=False)
    print '%s 最大相对误差: %e' % (param_name, rel_error(param_grad_num, grads[param_name]))


from nn.classifier_trainer import ClassifierTrainer
# from nn.classifiers.neural_net import init_two_layer_model
model = first_one_layer_model()

trainer = ClassifierTrainer()
# 这个地方是自己手造的数据，量不大，所以其实sample_batches就设为False了，直接全量梯度下降
best_model, loss_history, _, _ = trainer.train(x, y, x, y,
                                             model, two_layer_net,
                                             reg=0.001,
                                             learning_rate=1e-1, momentum=0.0, learning_rate_decay=1,
                                             update='sgd', sample_batches=False,
                                             num_epochs=100,
                                             verbose=False)
print 'Final loss with vanilla SGD: %f' % (loss_history[-1], )









from nn.data_utils import load_CIFAR10


def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000):
    cifar10_dir='nn/datasets/cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
    mask = range(num_training, num_training+num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    X_train = X_train[range(num_training)]
    y_train = y_train[range(num_training)]
    X_test = X_test[range(num_test)]
    y_test = y_test[range(num_test)]

    # 去均值
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image

    # 调整维度
    X_train= X_train.reshape(num_training, -1)
    X_test = X_test.reshape(num_test, -1)
    X_val = X_val.reshape(num_validation, -1)
    return X_train, y_train, X_val, y_val, X_test, y_test

X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data()
print 'Train data shape: ', X_train.shape
print 'Train labels shape: ', y_train.shape
print 'Validation data shape: ', X_val.shape
print 'Validation labels shape: ', y_val.shape
print 'Test data shape: ', X_test.shape
print 'Test labels shape: ', y_test.shape


from nn.classifiers.neural_net import init_two_layer_model
from nn.classifier_trainer import ClassifierTrainer

model = init_two_layer_model(32*32*3, 100, 10) # input size, hidden size, number of classes
trainer = ClassifierTrainer()
best_model, loss_history, train_acc, val_acc = trainer.train(X_train, y_train, X_val, y_val,
                                             model, two_layer_net,
                                             num_epochs=5, reg=1.0,
                                             momentum=0.9, learning_rate_decay = 0.95,
                                             learning_rate=1e-5, verbose=True)
