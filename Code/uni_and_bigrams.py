#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 21:48:55 2020

@author: qinwenna
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn import metrics

def vectorize(corpus, min_df=1, ngram=(1,2), max_features=None):
    # generate features
    vectorizer = TfidfVectorizer(stop_words='english', 
                                 min_df = min_df,
                                 ngram_range=ngram,
                                 max_features=max_features)
    X = vectorizer.fit_transform(corpus)
    X_arr = X.toarray()    
    
    # put into dataframe
    col_ind = vectorizer.get_feature_names()
    data = pd.DataFrame(X_arr, columns=col_ind)

    return vectorizer, data    
    
def random_forest(X_train, y_train):
    clf = RandomForestClassifier().fit(X_train, y_train)
    return clf

def linear_svm(X_train, y_train):
    clf = svm.SVC(kernel='linear', C=1, 
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
    
    return train_acc, val_acc, train_time, pred_time
         
         
        
def k_fold_cv(K, data):
    '''

    Parameters
    ----------
    K : int
        number of folds
    data : train + validation
           a pandas dataframe containing texts and stars

    Returns
    -------
    each numpy array is the results for a model, 
    with K rows and columns corresponding to training acc, validation acc,
    training time, prediction time
    '''
    kf = KFold(n_splits=K)
    
    res = {}
    rf_all = []
    lr_all = []
    nb_all = []
    
    for train_index, test_index in kf.split(data):
         train, val = data.iloc[train_index,:], data.iloc[test_index,:]
         
         # vectorize texts
         vect, X_train = vectorize(list(train['text']), max_features=100)
         y_train = train['stars']
         X_val = pd.DataFrame(vect.transform(list(val['text'])).toarray(),
                     columns = vect.get_feature_names())
         y_val = val['stars']
         # fit models and predict
         rf_res = pipe(random_forest, X_train, y_train, X_val, y_val)
         rf_all.append(rf_res)
         lr_res = pipe(log_reg, X_train, y_train, X_val, y_val)
         lr_all.append(lr_res)
         nb_res = pipe(mult_nb, X_train, y_train, X_val, y_val)
         nb_all.append(nb_res)
         
    res['rf'] = np.array(rf_all)
    res['lr'] = np.array(lr_all)
    res['nb'] = np.array(nb_all)
    
    return res

os.chdir('YelpData')
train = pd.read_csv('training.csv')
val = pd.read_csv('validation.csv')
data = pd.concat([train,val], axis=0)
K = 5
res_dict = k_fold_cv(K, data)


'''
vect, X_train = vectorize(list(train['text']), ngram=(2,2), max_features=1000)
y_train = train['stars']
X_val = pd.DataFrame(vect.transform(list(val['text'])).toarray(),
                     columns = vect.get_feature_names())
y_val = val['stars']

print("------Random Forest------")
rf_res = pipe(random_forest, X_train, y_train, X_val, y_val)
print("------Logistic Regression------")
lr_res = pipe(log_reg, X_train, y_train, X_val, y_val)
print("------Naive Bayes------")
nb_res = pipe(mult_nb, X_train, y_train, X_val, y_val)
print("------SVM------")
#svm_res = pipe(linear_svm, X_train, y_train, X_val, y_val)

'''