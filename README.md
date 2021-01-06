# 10701-Final-Project 

yezheng, ziyanl, wennaq
Sentiment Analysis: Predict Yelp Ratings from User Reviews

A repository for CMU 10701 Machine Learning Project.

Given Yelp Dataset, this project aims to determine factors that influence whether a review is positive or negative and construct models to predict the ratings of reviews based on texts and given attributes.

A detailed description of this project and the results can be found in the 10701-Final-Report.

Packages Used:
- numpy
- pandas
- sklearn
- nltk
- keras
- matplotlib

File Description:
- main_stemmed: run experiments with the vectorize function from bow_stemmed.py, in which BoW 2 and BoW 5 features could be generated;

- main_adj: run experiments with the vectorize function from bow_adj.py, which corresponds to BoW 3 features;

- basic_tfidf: no stemming/lemmatization, corresponding to BoW 1 features;

- bow_stemmed: contain functions for vectorization with stemming and computing top m correlated words;

- bow_lemmatized: contain functions for vectorization with lemmatization and computing top m correlated words; can be imported into main_stemmed for BoW 4 features;

- models: contains the simple models and a pipeline function that fits the model and returns accuracies;

- preprocess_reviews: preprocess all reviews with lemmatization and export the results to a csv file for future use;

- rnn: read lemmaztized reviews, creates word embedding matrices, and run LSTM; current hyperparameters in the file produced the plot in final report;

- uni_and_bigrams: can perform kfold cross validation for linear models;

- plot_corr: generate wordclouds presented in the report;

- eda: initial exploratory data analysis;

The GloVe word embeddings used for LSTM can be downloaded at: https://nlp.stanford.edu/projects/glove/
