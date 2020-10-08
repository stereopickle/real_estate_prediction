#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 16:56:36 2020

@author: stereopickles
"""

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import rcParams

import seaborn as sns

import folium
import json
from urllib.request import urlopen
import pickle

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX

from itertools import product

from sklearn.metrics import mean_squared_error as MSE
from math import sqrt

from fbprophet import Prophet
from fbprophet.plot import add_changepoints_to_plot


def run_dickyey_fuller(series, title):
    result = adfuller(series)
    p = result[1]
    if p < 0.05:
        print(f'Null Rejected (p = {round(p, 4)}). {title} time series is stationary')
    else: 
        print(f'Failed to reject the null (p = {round(p, 4)}). {title} time series is not stationary')


def select_geodata(values):
    ''' return geoJSON with only zipcodes in values '''
    geourl = 'https://raw.githubusercontent.com/OpenDataDE/State-zip-code-GeoJSON/master/ny_new_york_zip_codes_geo.min.json'
    with urlopen(geourl) as url:
        geodata = json.load(url)
    
    geo_feats = []
    for i in range(len(geodata['features'])):
        if geodata['features'][i]['properties']['ZCTA5CE10']  in values:
            geo_feats.append(geodata['features'][i])
            
    geodata_sel = {}
    geodata_sel['type'] = geodata['type']
    geodata_sel['features'] = geo_feats
    return geodata_sel


def plot_coropleth(df, columns, title, geodata_sel = {}):
    ''' return coropleth map''' 
    m = folium.Map(location=[40.73, -73.79])
    folium.Choropleth(
        geo_data=geodata_sel,
        name='choropleth',
        data=df,
        columns=columns,
        key_on='properties.ZCTA5CE10',
        fill_color='BuPu',
        fill_opacity=0.9,
        line_weight=1,
        legend_name=title, 
    ).add_to(m)
    
    folium.LayerControl().add_to(m)
    
    return m


def plot_basic(series, xlabel = 'year', ylabel = '', title = ''):
    ''' return basic time series plot ''' 
    plt.figure(figsize = (6, 4))
    series.plot()
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.show()

def plot_hist(series, xlabel = 'year', ylabel = 'frequency', 
              title = '', bins = 20):
    ''' return basic time series plot ''' 
    plt.figure(figsize = (6, 4))
    series.hist(grid=False, bins = bins)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.show()

# plotting moving average and sd
def plot_moving_avg(series, window, zipcode = ''):
    rolling_mean = series.rolling(window = window).mean()
    rolling_sd = series.rolling(window = window).std()
    plt.figure(figsize = (6, 4))
    plt.plot(series, label = 'Actual Values', lw = 2)    
    plt.plot(rolling_mean, '--', label = 'Rolling Mean', lw = 2)
    plt.plot(rolling_sd, '--', label = 'Rolling St. Dev', lw = 2)
    plt.legend()
    plt.ylabel('percent increase')
    plt.title(f'{window} months rolling trend ({zipcode})')
    plt.show()
    

def plot_decomposition(series, zipcode = ''):
    rcParams['figure.figsize'] = 10,4
    seasonal_decompose(series).plot()
    plt.xlabel(f'year ({zipcode})')
    plt.show()

def print_summary(series, zipcode):
    plot_basic(series, ylabel = f'percent increase ({zipcode})')
    plot_hist(series, ylabel = f'frequency ({zipcode})')
    plot_moving_avg(series, 24, zipcode)
    plot_decomposition(series, zipcode)
    summary = series.describe()
    print(f'[{zipcode} SUMMARY]')
    print(f"Mean: {round(summary['mean'], 2)}")
    print(f"Std: {round(summary['std'], 2)}")
    run_dickyey_fuller(series, zipcode)
    
def def_acf_pacf(series, zipcode):
    plt.figure(figsize = (8, 4))
    ax1 = plt.subplot(1, 2, 1)
    plot_acf(series, ax = ax1, alpha = 0.05)
    plt.title(f'ACF ({zipcode})')
    ax2 = plt.subplot(1, 2, 2)
    plot_pacf(series, ax = ax2, alpha = 0.05)
    plt.title(f'PACF ({zipcode})')
    plt.show()
    
    
def find_sarima_param(df, max_range = 2, 
                      p = range(0, 2), 
                      q = range(0, 2),
                      d = range(0, 2),
                      s = 12, thresh = 500):
    '''find the best sarima params'''
    
    pdq = list(product(p, d, q))
    spdq = [x + (s,) for x in pdq]
    min_param = [None, None, thresh]

    for param in pdq:
        for sparam in spdq:
            try:
                mod = SARIMAX(df, order=param,
                              seasonal_order=sparam)
                results = mod.fit()

                if results.aic < min_param[2]: 
                    min_param[0] = param
                    min_param[1] = sparam
                    min_param[2] = results.aic
            except:
                continue
    print(f'pdq: {min_param[0]}, PDQS: {min_param[1]} - AIC: {round(min_param[2], 2)}')
    return min_param[0], min_param[1]

def RMSE(y_true, y_pred):
    return sqrt(MSE(y_true, y_pred))

def get_RMSE(data, model, term = 40, show = True):
    pred = model.get_prediction(-term)
    pred_ci = pred.conf_int()
    sel_dat = data[-term:]
    rmse = RMSE(sel_dat, pred.predicted_mean)
    if show:
        ax = data.plot(label='observed', figsize=(8, 6))
        pred.predicted_mean.plot(ax=ax, label='predicted')
        ax.fill_between(pred_ci.index,
                        pred_ci.iloc[:, 0],
                        pred_ci.iloc[:, 1], color='k', alpha=0.1)
        ax.set_xlabel('Date')
        ax.set_ylabel('Percent Increase')
        
        plt.legend(loc='best')
        plt.show()
    return rmse

def rolling_forecast(df, pdq, SPDQ, term = 40):
    train = df[:-term]
    test = df[-term:]
    hist = list(train)
    pred = list()
    for term in range(len(test)):
        model = SARIMAX(hist, order = pdq, 
                        seasonal_order = SPDQ)
        fit = model.fit(disp=0)
        pred.append(fit.forecast()[0])
        hist.append(test[term]) 
    return train, test, hist, pred

def test_RMSE(df, pdq, SPDQ, term = 40, show = True):
    ''' get RMSE for test set '''
    _, test, _, pred = rolling_forecast(df, pdq, SPDQ, term = term)
    
    if show:
        plt.figure(figsize=(8, 6))
        plt.plot(df, label='true')
        plt.plot(test.index, pred, label='forecast')
        plt.xlabel('Date')
        plt.ylabel('Percent Increase')        
        plt.legend(loc='best')
        plt.title('rolling forecasting on test set')
        plt.show()
    return RMSE(test, pred)

def fb_prophet_forecast(df0, changepoint_scale = 1, term = 40):
    df = df0.copy()
    df.columns = ['ds', 'y']
    train = df[:-term]
    test = df[-term:]
    model = Prophet(changepoint_prior_scale = changepoint_scale,
                    yearly_seasonality = True, 
                    daily_seasonality = False,
                    weekly_seasonality = False,
                    interval_width = 0.95)
    
    model.add_country_holidays(country_name = 'US')
    model.fit(train)
    future = model.make_future_dataframe(periods = term, 
                                               freq = 'MS')
    forecast = model.predict(future)
    rmse = RMSE(test['y'], list(forecast['yhat'][-term:]))
    return rmse, forecast, model

        
def fbp_plot(forecast, model, changepoint = False):

    fig = model.plot(forecast, xlabel = 'date', 
               ylabel = 'percent increase', figsize = (8, 6))
    if changepoint:
        a = add_changepoints_to_plot(fig.gca(), model, forecast)
    plt.show()
    model.plot_components(forecast, figsize = (8, 6))
    plt.show()
    
    