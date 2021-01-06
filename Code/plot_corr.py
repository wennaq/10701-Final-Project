#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 10:24:12 2020

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
from bow_stemmed import vectorize

from wordcloud import WordCloud, ImageColorGenerator
import matplotlib.pyplot as plt

def plot_common_words(D):
        
    wordcloud = WordCloud(width=2000, height=1200, 
                          colormap='Pastel1', random_state=1).\
        generate_from_frequencies(D)
    fig = plt.figure(figsize=(30, 10), facecolor="white")
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    plt.title('Top 500 Words Most Correlated with Stars', fontsize=50)
    plt.tight_layout(pad=0)

    return 

cwd = os.getcwd()
train = pd.read_csv(cwd+'/YelpData/training.csv')
val = pd.read_csv(cwd+'/YelpData/validation.csv')
# vectorize data using bag of words model
vect, X_train = vectorize(list(train['text']))
y_train = train['stars']
m = 500
col_ind = vect.get_feature_names()
corr = {}
for i in range(X_train.shape[1]):
    col = X_train.iloc[:,i]
    word = col_ind[i]
    cur_corr = abs(col.corr(y_train, method="pearson"))
    corr[word] = cur_corr
    
sorted_corr = pd.DataFrame([(k,v) for k, v in sorted(corr.items(), 
                            key=lambda item: item[1], reverse=True)],
                           columns=["word", "corr"])
topm = list(sorted_corr['word'][:m])
topcorr = {}
for word in topm:
    topcorr[word] = corr[word]
    


plot_common_words(topcorr)

