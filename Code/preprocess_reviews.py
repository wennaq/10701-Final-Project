#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 08:51:46 2020

@author: qinwenna
"""
import os
import pandas as pd
import numpy as np
import re
import nltk
import contractions
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt

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

def lemmatize(tokens):
    pos_tags = nltk.tag.pos_tag(tokens)
    wordnet_pos = [(word, get_wordnet_pos(pos_tag)) for 
                   (word, pos_tag) in pos_tags]
    wnl = WordNetLemmatizer()
    lemmatized = [wnl.lemmatize(word, tag) if tag != None else word 
                  for (word, tag) in wordnet_pos]
    return lemmatized

def tokenize(sen, sw):
    sen_tokens = word_tokenize(sen)
    # remove stopwords
    sen_tokens_nosw = [word for word in sen_tokens if not word in sw]
    
    # Lemmatize
    lemmatized = lemmatize(sen_tokens_nosw)
    
    return lemmatized

def preprocess(sen, sw):
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

    lemmatized = tokenize(sentence,sw)
    
    sentence = ' '.join(lemmatized)
    
    return sentence



def preprocess_text(sen, sw):

    # convert to lower case
    sentence = sen.lower()
    
    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)

    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)
    
    # Remove newline
    sentence = " ".join(sentence.splitlines())
    
    sen_tokens = word_tokenize(sentence)

    # remove stopwords
    sen_tokens_nosw = [word for word in sen_tokens if not word in sw]
    
    pos_tags = nltk.tag.pos_tag(sen_tokens_nosw)
    wordnet_pos = [(word, get_wordnet_pos(pos_tag)) for 
                   (word, pos_tag) in pos_tags]
    wnl = WordNetLemmatizer()
    lemmatized = [wnl.lemmatize(word, tag) for (word, tag) in wordnet_pos]
    
    sentence = ' '.join(lemmatized)

    return sentence



cwd = os.getcwd()
test = pd.read_csv(cwd+'/YelpData/testing.csv')
all_stopwords = stopwords.words('english')
#all_stopwords.remove('not')
headers = ['stars', 'text', 'lemmatized']
test_df = test[['stars', 'text']]
test_df = test_df.reindex(columns=headers)
for i in range(test_df.shape[0]):
    text = test_df.iloc[i,1]
    lemmatized_str = preprocess(text, all_stopwords)
    test_df.iloc[i,2] = lemmatized_str
    
test_df.to_csv("test_lemmatized_2.csv")

'''
cwd = os.getcwd()
train = pd.read_csv(cwd+'/YelpData/training.csv')
val = pd.read_csv(cwd+'/YelpData/validation.csv')
all_stopwords = stopwords.words('english')
#all_stopwords.remove('not')
headers = ['stars', 'text', 'lemmatized']
train_df = train[['stars', 'text']]
train_df = train_df.reindex(columns=headers)
for i in range(train_df.shape[0]):
    text = train_df.iloc[i,1]
    lemmatized_str = preprocess(text, all_stopwords)
    train_df.iloc[i,2] = lemmatized_str
    
train_df.to_csv("train_lemmatized_2.csv")

val_df = train[['stars', 'text']]
val_df = train_df.reindex(columns=headers)
for i in range(val_df.shape[0]):
    text = val_df.iloc[i,1]
    lemmatized_str = preprocess(text, all_stopwords)
    val_df.iloc[i,2] = lemmatized_str

val_df.to_csv("val_lemmatized_2.csv")
'''

