#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 10:23:35 2020

@author: qinwenna
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import nltk
import contractions
import time
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn import svm
from sklearn.model_selection import KFold
from models import pipe
from models import random_forest
from models import log_reg
from models import mult_nb
from models import linear_svm

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None
    
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

# Note: wordnet does not have pronouns
def lemmatize(tokens):
    pos_tags = nltk.tag.pos_tag(tokens)
    wordnet_pos = [(word, get_wordnet_pos(pos_tag)) for 
                   (word, pos_tag) in pos_tags]
    wnl = WordNetLemmatizer()
    lemmatized = [wnl.lemmatize(word, tag) if tag != None else word 
                  for (word, tag) in wordnet_pos]
    return lemmatized

def tokenize(sen):
    sen_tokens = word_tokenize(sen)
    
    # Lemmatize
    lemmatized = lemmatize(sen_tokens)
    
    return lemmatized

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
'''
cwd = os.getcwd()
train = pd.read_csv(cwd+'/YelpData/training.csv')
val = pd.read_csv(cwd+'/YelpData/validation.csv')
vect, X_train = vectorize(list(train['text']))
y_train = train['stars']
X_val = pd.DataFrame(vect.transform(list(val['text'])).toarray(),
                     columns = vect.get_feature_names())
y_val = val['stars']
m = 500
col_ind = vect.get_feature_names()
X_train_m, X_val_m = topm_correlation(X_train, y_train, 
                                      X_val, col_ind, m)

print("------Random Forest------")
rf_res = pipe(random_forest, X_train_m, y_train, X_val_m, y_val)
print("------Logistic Regression------")
lr_res = pipe(log_reg, X_train_m, y_train, X_val_m, y_val)
print("------Naive Bayes------")
nb_res = pipe(mult_nb, X_train_m, y_train, X_val_m, y_val)
print("------SVM------")
svm_res = pipe(linear_svm, X_train_m, y_train, X_val_m, y_val)

'''
