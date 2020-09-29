#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 17:20:18 2020

@author: stereopickles
"""

from statsmodels.tsa.stattools import adfuller


def run_dickyey_fuller(series):
    result = adfuller(series)
    p = result[1]
    if p < 0.05:
        print(f'Null Rejected (p = {round(p, 4)}). This series is stationary')
    else: 
        print(f'Failed to reject the null (p = {round(p, 4)}). This series is not stationary')
