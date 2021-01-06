#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 12:42:46 2020

@author: qinwenna
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
from models import pipe
from models import random_forest
from models import log_reg
from models import mult_nb
from models import linear_svm
from bow_stemmed import vectorize
from bow_stemmed import topm_correlation
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import log_loss
from sklearn import svm
from sklearn.metrics import plot_confusion_matrix

cwd = os.getcwd()
train = pd.read_csv(cwd+'/YelpData/training.csv')
val = pd.read_csv(cwd+'/YelpData/validation.csv')
test = pd.read_csv(cwd+'/YelpData/testing.csv')
# vectorize data using bag of words model
vect, X_train = vectorize(list(train['text']), ngram=(1,1))
y_train = train['stars']
X_val = pd.DataFrame(vect.transform(list(val['text'])).toarray(),
                     columns = vect.get_feature_names())
y_val = val['stars']
X_test = pd.DataFrame(vect.transform(list(test['text'])).toarray(),
                     columns = vect.get_feature_names())
y_test = test['stars']

# select the top m correlated words
m = 500
col_ind = vect.get_feature_names()
X_train_m, X_val_m = topm_correlation(X_train, y_train, 
                                      X_val, col_ind, m)
X_train_m, X_test_m = topm_correlation(X_train, y_train,
                                       X_test, col_ind, m)
# fit model
#clf = LogisticRegression(C=0.5, solver='sag', multi_class="multinomial")
#clf_fit = clf.fit(X_train_m, y_train)
clf_svm = svm.LinearSVC(C=0.3).fit(X_train_m, y_train)
# compute accuracy
train_pred = clf.predict(X_train_m)
val_pred = clf.predict(X_val_m)
train_acc = metrics.accuracy_score(y_train, train_pred)
print("train acc: "+str(train_acc))
val_acc = metrics.accuracy_score(y_val, val_pred)
print("val acc: "+str(val_acc))
# plot confusion matrix
fig, ax = plt.subplots(figsize=(20, 20))
plot_confusion_matrix(clf, X_val_m, y_val, normalize='true',
                       cmap=plt.cm.Blues)

test_pred = clf.predict(X_test_m)
test_acc = metrics.accuracy_score(y_test, test_pred)

'''
print("------Random Forest------")
rf_res = pipe(random_forest, X_train_m, y_train, X_val_m, y_val)
print("------Logistic Regression------")
lr_res = pipe(log_reg, X_train_m, y_train, X_val_m, y_val)
print("------Naive Bayes------")
nb_res = pipe(mult_nb, X_train_m, y_train, X_val_m, y_val)
print("------SVM------")
svm_res = pipe(linear_svm, X_train_m, y_train, X_val_m, y_val)



clf = rf_res[0]
feat_imp = pd.Series(clf.feature_importances_,
                        index=X_train_df.columns.values)
feat_imp = feat_imp.sort_values(ascending=False)
'''

