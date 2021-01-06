#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code source: https://stackabuse.com/python-for-nlp-movie-sentiment-analysis-using-deep-learning-in-keras/
"""
import os
import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt

from numpy import array
from numpy import asarray
from numpy import zeros
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten
from keras.layers import GlobalMaxPooling1D
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.layers.recurrent import LSTM
from keras.utils import to_categorical


cwd = os.getcwd()
train = pd.read_csv(cwd+'/YelpData/train_lemmatized_2.csv')
val = pd.read_csv(cwd+'/YelpData/val_lemmatized_2.csv')
X_train = list(train['lemmatized'])
y_train = list(train['stars'])
X_val = list(val['lemmatized'])
y_val = list(val['stars'])
    

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)
word2index = tokenizer.word_index
index2word = tokenizer.index_word

X_train = tokenizer.texts_to_sequences(X_train)
X_val = tokenizer.texts_to_sequences(X_val)

# Adding 1 because of reserved 0 index
vocab_size = len(tokenizer.word_index) + 1

maxlen = 100

X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_val = pad_sequences(X_val, padding='post', maxlen=maxlen)

embeddings_dictionary = dict()
glove_file = open('/Users/qinwenna/Desktop/10701/FinalProject/glove.twitter.27B/glove.twitter.27B.100d.txt', encoding="utf8")

for line in glove_file:
    records = line.split()
    word = records[0]
    vector_dimensions = asarray(records[1:], dtype='float32')
    embeddings_dictionary [word] = vector_dimensions
glove_file.close()

embedding_matrix = zeros((vocab_size, 100))
for word, index in tokenizer.word_index.items():
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector

model = Sequential()
embedding_layer = Embedding(vocab_size, 100, weights=[embedding_matrix], 
                            input_length=maxlen , trainable=False)
model.add(embedding_layer)
model.add(LSTM(128, dropout=0.2))
model.add(Dense(6, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', 
              metrics=['accuracy'])


history = model.fit(X_train, to_categorical(y_train), batch_size=128, 
                    epochs=30, verbose=1, 
                    validation_data=(X_val,to_categorical(y_val)))

score = model.evaluate(X_val, to_categorical(y_val), verbose=1)

plt.figure(figsize=(7,5))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','val'], loc='best')
plt.show()

plt.figure(figsize=(7,5))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','val'], loc='best')
plt.show()
