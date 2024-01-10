# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 19:36:16 2023

@author: Elijah
"""
from gplearn.functions import _Function
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error
import numpy as np

def first_accumulation(y, t, n):
    h = np.diff(t).astype(np.float64)
    h=np.divide(h,np.mean(h))
    t=np.cumsum(np.concatenate((np.array([1]),h)))
    x_1=np.cumsum(np.concatenate((np.array([1]),h))*y)
    return x_1,t

def _equation_str(tmp):
    """Overloads `print` output of the object to resemble a LISP tree."""
    terminals = [0]
    output = ''
    for i, node in enumerate(tmp.program):
        if isinstance(node, _Function):
            terminals.append(node.arity)
            if node.name=='add':
                name='np.add'
            if node.name=='sub':
                name='np.subtract'
            if node.name=='mul':
                name='np.multiply'
            if node.name=='div':
                name='np.divide'
            if node.name=='sqrt':
                name='np.sqrt'
            if node.name=='log':
                name='np.log'
            if node.name=='neg':
                name='np.negative'
            if node.name=='abs':
                name='np.abs'
            if node.name=='max':
                name='np.maximum'
            if node.name=='min':
                name='np.minimum'
            if node.name=='sin':
                name='np.sin'
            if node.name=='cos':
                name='np.cos'
            if node.name=='tan':
                name='np.tan'
            output += name + '('
        else:
            if isinstance(node, int):
                if tmp.feature_names is None:
                    output += 'X%s' % node
                else:
                    output += tmp.feature_names[node]
            else:
                output += '%.3f' % node
            terminals[-1] -= 1
            while terminals[-1] == 0:
                terminals.pop()
                terminals[-1] -= 1
                output += ')'
            if i != len(tmp.program) - 1:
                output += ', '
    return output

def _print_error(y,y_pre,train_size,test_size):
    print('The fitting MAE',
          mean_absolute_error(y[0:train_size], y_pre[0:train_size]), '\n'
          'The fitting MAPE',
          mean_absolute_percentage_error(y[0:train_size], y_pre[0:train_size]),'\n'
          'The forecasting MAE',
          mean_absolute_error(y[train_size:], y_pre[train_size:train_size+test_size]), '\n'
          'The forecasting RMSE',
          np.sqrt(mean_squared_error(y[train_size:], y_pre[train_size:train_size+test_size])), '\n'
          'The forecasting MAPE',
          mean_absolute_percentage_error(y[train_size:], y_pre[train_size:train_size+test_size])
      )

def NIP_accumulation(y, t, n, r):
    NIP_matrix_r=np.zeros((n,n))
    for i in range(n):
        for j in range(i+1):
            NIP_matrix_r[i][j]=np.power(r,t[i]-t[j])
    xr=NIP_matrix_r@y
    return xr,NIP_matrix_r

def replace_function(substring,input_string):
    if substring=='subtract':
        replacement='-'
    if substring=='add':
        replacement='+'
    if substring=='multiply':
        replacement='*'
    if substring=='divide':
        replacement='/'
    if substring in input_string:
        position = input_string.find(substring)
        tmp=0
        ind=position-1
        for i in input_string[position:]:
            ind=ind+1 
            if i=='(':
                tmp=tmp+1
            if i==')':
                tmp=tmp-1
            if i==',' and tmp==1:
                output_string=input_string[:ind] + replacement + input_string[ind+1:]
                break
        output_string=output_string[:position]+output_string[position+len(substring):]
    return output_string

def replace_string(input_string):
    input_string=input_string.replace('np.','')
    replace_string = ['add','subtract','multiply','divide']
    for substring in replace_string:
        while substring in input_string:
            input_string=replace_function(substring,input_string)
    return input_string



