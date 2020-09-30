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

from SCRIPT.eval_tools import *

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