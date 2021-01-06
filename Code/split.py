#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 10:08:36 2020

@author: qinwenna
"""
import pandas as pd
from sklearn.model_selection import train_test_split

yelp_train = pd.read_csv('Yelp_train.csv')

train_all, test = train_test_split(yelp_train, test_size=0.2, random_state=42)
train, val = train_test_split(train_all, test_size=0.25, random_state=42)

train.to_csv("training.csv")
val.to_csv("validation.csv")
test.to_csv("testing.csv")