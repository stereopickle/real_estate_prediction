#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 22:11:59 2020

@author: stereopickles
"""
from SCRIPT.eval_tools import RMSE
import numpy as np
import matplotlib.pyplot as plt

def sequence_generator(series0, steps = 3):
    '''
    INPUT: time series as pd.Series and number of steps
    OUTPUT input list and output list
    '''
    series = series0.values
    inputs = list()
    outputs = list()
    for i in range(steps, len(series)):
        outputs.append(series[i])
        inputs.append(series[i-3:i])
    inputs = np.expand_dims(inputs, 2)
    return inputs, np.array(outputs)

def show_RMSE(df, y_test, y_pred, steps = 3, show = True):
    ''' plot RMSE for test set '''
    if show:
        tedf = df[-(40+steps):]
        plt.figure(figsize=(8, 6))
        plt.plot(df, label='true')
        plt.plot(tedf.index[steps:], y_pred, label='forecast')
        plt.xlabel('Date')
        plt.ylabel('Percent Increase')        
        plt.legend(loc='best')
        plt.title('forecast')
        plt.show()
    return RMSE(y_test, y_pred)