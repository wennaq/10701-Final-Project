#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 10:23:35 2020

@author: qinwenna

In this file, the TfidfVectorizer is used with stopwords being removed and 
max_features being set. Stemming was performed. 5-fold cv was used to evaluate 
models. Unable to run SVM with more than 300 features.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import nltk
import time
import contractions
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn import svm
from sklearn.model_selection import KFold

from models import pipe
from models import random_forest
from models import log_reg
from models import mult_nb
from models import linear_svm
    
def preprocess(sen):
    # expand contractions
    sentence = contractions.fix(sen)
    
    # convert to lower case
    sentence = sentence.lower()
    
    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)

    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)
    
    # Remove newline
    sentence = " ".join(sentence.splitlines())

    return sentence

def tokenize(sen):
    sen_tokens = word_tokenize(sen)
    
    # Stemming
    snow_stemmer = SnowballStemmer(language='english') 
    stemmed = [snow_stemmer.stem(word) for word in sen_tokens]
    
    return stemmed

def vectorize(corpus, ngram=(1,1)):
    # make stopwords consistent with preprocessing
    eng_stopwords = []
    for word in stopwords.words('english'):
        word = preprocess(word)
        word_tokens = tokenize(word)
        eng_stopwords.extend(word_tokens)
        
    
    # Bag of Words: generate features
    start = time.time()
    vectorizer = TfidfVectorizer(stop_words=eng_stopwords, min_df=0.001,
                                 ngram_range=ngram,
                                 preprocessor=preprocess,
                                 tokenizer=tokenize,
                                 analyzer='word')
    # extract features using training data
    X_train = vectorizer.fit_transform(corpus)
    X_train_arr = X_train.toarray()    
    stop = time.time()
    print("time elapsed:"+str(stop-start))
    col_ind = vectorizer.get_feature_names()
    X_train_df = pd.DataFrame(X_train_arr, columns=col_ind)
    
    return vectorizer, X_train_df
        
def topm_correlation(X_train_df, y_train, X_val_df, col_ind, m):
    corr = {}
    for i in range(X_train_df.shape[1]):
        col = X_train_df.iloc[:,i]
        word = col_ind[i]
        cur_corr = abs(col.corr(y_train, method="pearson"))
        corr[word] = cur_corr
        
    sorted_corr = pd.DataFrame([(k,v) for k, v in sorted(corr.items(), 
                                key=lambda item: item[1], reverse=True)],
                               columns=["word", "corr"])
    topm = list(sorted_corr['word'][:m])
    X_train_m = X_train_df[topm]
    X_val_m = X_val_df[topm]
    
    return X_train_m, X_val_m


def k_fold_cv(K, data):
    kf = KFold(n_splits=K)
    
    res = {}
    rf_all = []
    lr_all = []
    nb_all = []
    k = 0
    for train_index, test_index in kf.split(data):
        print("k  = "+str(k))
        train, val = data.iloc[train_index,:], data.iloc[test_index,:]
        
        # vectorize texts
        vect, X_train = vectorize(list(train['text']))
        y_train = train['stars']
        col_ind = vect.get_feature_names()
        X_val = pd.DataFrame(vect.transform(list(val['text'])).toarray(),
                    columns = col_ind)
        y_val = val['stars']
        # use only top m most correlated words
        m =  100
        X_train_m, X_val_m = topm_correlation(X_train, y_train, 
                                              X_val, col_ind, m)
        # fit models and predict
        print("------Random Forest------")
        rf_res = pipe(random_forest, X_train, y_train, X_val, y_val)
        rf_all.append(rf_res)
        print("------Logistic Regression------")
        lr_res = pipe(log_reg, X_train, y_train, X_val, y_val)
        lr_all.append(lr_res)
        print("------Naive Bayes------")
        nb_res = pipe(mult_nb, X_train, y_train, X_val, y_val)
        nb_all.append(nb_res)
        
        k += 1
         
    res['rf'] = np.array(rf_all)
    res['lr'] = np.array(lr_all)
    res['nb'] = np.array(nb_all)
    
    return res



'''
data = pd.concat([train,val], axis=0)
K = 5
res_dict = k_fold_cv(K, data)


linear = svm.SVC(kernel='linear', C=1, 
                  decision_function_shape='ovo').fit(X_train_m, y_train)
linear_pred = linear.predict(X_val_m)
accuracy_lin = metrics.accuracy_score(y_val, linear_pred)
'''