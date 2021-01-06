#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 14:04:38 2020

@author: qinwenna
"""

import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn import metrics


def random_forest(X_train, y_train):
    clf = RandomForestClassifier().fit(X_train, y_train)
    return clf

def linear_svm(X_train, y_train):
    #clf = svm.SVC(kernel='linear', C=1, 
    #                 decision_function_shape='ovo').fit(X_train, y_train)
    clf = svm.LinearSVC().fit(X_train, y_train)
    return clf

def rbf_svm(X_train, y_train):
    clf = svm.SVC(kernel='rbf', C=1, 
                     decision_function_shape='ovo').fit(X_train, y_train) 
    return clf

def log_reg(X_train, y_train):
    clf = LogisticRegression(solver='sag').fit(X_train, y_train)
    return clf


def mult_nb(X_train, y_train):
    clf = MultinomialNB().fit(X_train, y_train)
    return clf

def pipe(model, X_train, y_train, X_val, y_val):
    start = time.time()
    clf = model(X_train, y_train)
    stop = time.time()
    train_time = stop-start
    train_acc = metrics.accuracy_score(y_train, clf.predict(X_train))
    
    start = time.time()
    pred = clf.predict(X_val)
    stop = time.time()
    pred_time = stop-start
    val_acc = metrics.accuracy_score(y_val, pred)
    
    return clf, train_acc, val_acc, train_time, pred_time