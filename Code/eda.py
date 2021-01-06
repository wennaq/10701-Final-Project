#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 11:08:05 2020

@author: qinwenna
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

print(os.getcwd())
os.chdir('YelpData')
yelp_train = pd.read_csv('Yelp_train.csv')
desc = yelp_train.describe()
headers = yelp_train.columns.values
yelp_train.isnull().sum()
# 1912 missing data in sentiment_score
headers_num = ['stars', 'useful', 'funny', 'cool','longitude', 'latitude',
               'nchar', 'nword', 'sentiment_score']
yelp_train['stars'].plot.hist()
yelp_train['stars'].plot.box()
yelp_train['sentiment_score'].plot.box()
yelp_train[['useful', 'funny', 'cool']].plot.box()
yelp_train['longitude'].plot.box()
yelp_train['latitude'].plot.box()
cor = yelp_train[headers_num].corr()
cor.style.background_gradient(cmap='coolwarm').set_precision(2)