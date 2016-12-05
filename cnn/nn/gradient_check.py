#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from random import randrange

def eval_numerical_gradient(f, x, verbose=True, h=0.00001):
  """ 
  计算函数f在x处的数值梯度
  - f 是接受1个参数的函数
  - x 是需要计算梯度的点
  """

  fx = f(x) # 求函数值
  grad = np.zeros_like(x)
  # 遍历所有x的维度
  it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
  while not it.finished:

    # 计算x+h和x-h处的函数值
    ix = it.multi_index
    oldval = x[ix]
    x[ix] = oldval + h
    fxph = f(x)
    x[ix] = oldval - h
    fxmh = f(x)
    x[ix] = oldval

    # 根据公式求解梯度(斜率)
    grad[ix] = (fxph - fxmh) / (2 * h)
    if verbose:
      print ix, grad[ix]
    it.iternext() # 切到下个x的维度

  return grad


def eval_numerical_gradient_array(f, x, df, h=1e-5):
  """
  对于多维度的x(多维numpy array)，求解数值梯度
  """
  grad = np.zeros_like(x)
  it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
  while not it.finished:
    ix = it.multi_index

    oldval = x[ix]
    x[ix] = oldval + h
    pos = f(x)
    x[ix] = oldval - h
    neg = f(x)
    x[ix] = oldval

    grad[ix] = np.sum((pos - neg) * df) / (2 * h)
    it.iternext()
  return grad


def eval_numerical_gradient_blobs(f, inputs, output, h=1e-5):
  """
  Compute numeric gradients for a function that operates on input
  and output blobs.
  
  We assume that f accepts several input blobs as arguments, followed by a blob
  into which outputs will be written. For example, f might be called like this:

  f(x, w, out)
  
  where x and w are input Blobs, and the result of f will be written to out.

  Inputs: 
  - f: function
  - inputs: tuple of input blobs
  - output: output blob
  - h: step size
  """
  numeric_diffs = []
  for input_blob in inputs:
    diff = np.zeros_like(input_blob.diffs)
    it = np.nditer(input_blob.vals, flags=['multi_index'],
                   op_flags=['readwrite'])
    while not it.finished:
      idx = it.multi_index
      orig = input_blob.vals[idx]

      input_blob.vals[idx] = orig + h
      f(*(inputs + (output,)))
      pos = np.copy(output.vals)
      input_blob.vals[idx] = orig - h
      f(*(inputs + (output,)))
      neg = np.copy(output.vals)
      input_blob.vals[idx] = orig

      diff[idx] = np.sum((pos - neg) * output.diffs) / (2.0 * h)

      it.iternext()
    numeric_diffs.append(diff)
  return numeric_diffs


def eval_numerical_gradient_net(net, inputs, output, h=1e-5):
  return eval_numerical_gradient_blobs(lambda *args: net.forward(),
                                       inputs, output, h=h)


def grad_check_sparse(f, x, analytic_grad, num_checks):
  """
  随机取x的一些维度，然后比对解析梯度和数值梯度的值
  """
  h = 1e-5

  x.shape
  for i in xrange(num_checks):
    ix = tuple([randrange(m) for m in x.shape])

    # 计算x+h和x-h处的函数值
    oldval = x[ix]
    x[ix] = oldval + h
    fxph = f(x)
    print type(fxph), type(fxmh)
    x[ix] = oldval - h
    fxmh = f(x)
    x[ix] = oldval

    # 根据公式求解梯度(斜率)
    grad_numerical = (fxph - fxmh) / (2 * h)
    grad_analytic = analytic_grad[ix]
    rel_error = abs(grad_numerical - grad_analytic) / (abs(grad_numerical) + abs(grad_analytic))
    print 'numerical: %f analytic: %f, relative error: %e' % (grad_numerical, grad_analytic, rel_error)

