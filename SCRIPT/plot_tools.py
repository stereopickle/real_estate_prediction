#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 16:56:36 2020

@author: stereopickles
"""

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import folium
import json
from urllib.request import urlopen
import pickle


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
    series.plot()
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.show()

def plot_hist(series, xlabel = 'year', ylabel = 'frequency', 
              title = '', bins = 20):
    ''' return basic time series plot ''' 
    series.hist(grid=False, bins = bins)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.show()

# plotting moving average and sd
def plot_moving_avg(series, window):
    rolling_mean = series.rolling(window = window).mean()
    rolling_sd = series.rolling(window = window).std()
    plt.figure(figsize = (8, 5))
    plt.plot(rolling_mean, '--', label = 'Rolling Mean')
    plt.plot(rolling_sd, '--', label = 'Rolling St. Dev')
    plt.plot(series, label = 'Actual Values')
    plt.legend()
    plt.ylabel('percent increase')
    plt.title(f'{window} months rolling trend')
    plt.show()
    
