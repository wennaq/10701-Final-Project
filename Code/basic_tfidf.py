'''
In this file, the TfidfVectorizer is used with stopwords being removed and 
max_features being set. No stemming/lemmeatization was performed. 5-fold cv
was used to evaluate models. 
'''
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

def vectorize(corpus, min_df=1, ngram=(1,1), max_features=None):
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

    return vectorizer, data, X
    
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
    models = {}
    rf_all = []
    rf_m = []
    lr_all = []
    lr_m = []
    nb_all = []
    nb_m = []
    svm_all = []
    svm_m = []
    vect_all =[]
    
    for train_index, test_index in kf.split(data):
        train, val = data.iloc[train_index,:], data.iloc[test_index,:]
         
        # vectorize texts
        vect, X_train = vectorize(list(train['text']), max_features=1000)
        vect_all.append(vect)
        y_train = train['stars']
        X_val = pd.DataFrame(vect.transform(list(val['text'])).toarray(),
                    columns = vect.get_feature_names())
        y_val = val['stars']
        # fit models and predict
        print("rf")
        rf_res = pipe(random_forest, X_train, y_train, X_val, y_val)
        rf_m.append(rf_res[0])
        rf_all.append(rf_res[1:])
        print("lr")
        lr_res = pipe(log_reg, X_train, y_train, X_val, y_val)
        lr_m.append(lr_res[0])
        lr_all.append(lr_res[1:])
        print("nb")
        nb_res = pipe(mult_nb, X_train, y_train, X_val, y_val)
        nb_m.append(nb_res[0])
        nb_all.append(nb_res[1:])
        print("svm")
        svm_res = pipe(linear_svm, X_train, y_train, X_val, y_val)
        svm_m.append(svm_res[0])
        svm_all.append(svm_res[1:])
         
    res['rf'] = np.array(rf_all)
    res['lr'] = np.array(lr_all)
    res['nb'] = np.array(nb_all)
    res['svm'] = np.array(svm_all)
    models['rf'] = rf_m
    models['lr'] = lr_m
    models['nb'] = nb_m
    
    return res, models
'''
os.chdir('YelpData')
train = pd.read_csv('training.csv')
val = pd.read_csv('validation.csv')

vect, X_train = vectorize(list(train['text']), ngram=(1,1), max_features=1000)
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
svm_res = pipe(linear_svm, X_train, y_train, X_val, y_val)
'''