#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 13:18:42 2020

@author: qinwenna
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 12:42:46 2020

@author: qinwenna
"""
import os
import pandas as pd
from models import pipe
from models import random_forest
from models import log_reg
from models import mult_nb
from models import linear_svm
from bow_adj import vectorize

cwd = os.getcwd()
train = pd.read_csv(cwd+'/YelpData/training.csv')
val = pd.read_csv(cwd+'/YelpData/validation.csv')
# vectorize data using bag of words model
vect, X_train = vectorize(list(train['text']))
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
clf = rf_res[0]
feat_imp = pd.Series(clf.feature_importances_,
                        index=X_train_df.columns.values)
feat_imp = feat_imp.sort_values(ascending=False)
'''

