#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


def init_two_layer_model(input_size, hidden_size, output_size):
    model = {}
    model["w1"] = np.random.randn(input_size, hidden_size)\
                  * np.sqrt(2.0/(input_size*hidden_size))
    model["b1"] = np.zeros(hidden_size)
    model["w2"] = np.random.randn(hidden_size, output_size)\
                    * np.sqrt(2.0/(hidden_size*output_size))
    model["b2"] = np.zeros(output_size)
    return model


def two_layer_net(inputs, model, labels=None, reg=0.0, verbose=False):
    # input是输入矩阵,其中列数是输出神经元个数,相当于分成多少个特征输入,一行是一个sample
    # labels是正确的output标准, 默认为none

    w1,b1,w2,b2 = model["w1"],model["b1"],model["w2"],model["b2"]
    N,D=inputs.shape
    scores = None

    hidden_activation = np.maximum( inputs.dot(w1) + b1, 0)
    if verbose: print "Layer 1 result shape: " + str(hidden_activation.shape)

    # softmax 之前得分
    scores = hidden_activation.dot(w2) + b2
    if verbose: print "Layer 2 result shape: " + str(scores.shape)

    if labels is None:
        return scores

    loss = 0
    # tricks , minus all scores by the highest score to keep variance
    scores = scores - np.expand_dims( np.amax(scores, axis=1), axis=1)
    exp_scores = np.exp(scores)
    # softmax得分
    probs = exp_scores/ np.sum(exp_scores, axis=1, keepdims=True)

    loss = np.sum(-scores[range(len(labels)), labels] + np.log(np.sum(exp_scores, axis=1)))/N
    loss += 0.5* reg * (np.sum( exp_scores, axis=1, keepdims=True))

    # 下来计算梯度
    grads = {}
    delta_scores = probs
    delta_scores[range(N), labels] -= 1
    delta_scores /= N
    grads["w2"] = hidden_activation.T.dot(delta_scores)
    grads["b2"] = np.sum( delta_scores, axis=0)

    delta_hidden = delta_scores.dot(w2.T)
    delta_hidden[hidden_activation<=0] = 0

    grads["w1"] = inputs.T.dot(delta_hidden)
    grads["b1"] = np.sum(delta_hidden, axis= 0)

    grads["w2"] = reg * w2
    grads["w1"] = reg * w1
    return loss,grads