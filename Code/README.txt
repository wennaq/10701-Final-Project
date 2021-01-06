# 10701GroupProject yezheng, ziyanl, wennaqmain_stemmed: run experiments with the vectorize function from bow_stemmed.py, in which BoW 2 and BoW 5 features could be generated;
main_adj: run experiments with the vectorize function from bow_adj.py, which corresponds to BoW 3 features;
basic_tfidf: no stemming/lemmatization, corresponding to BoW 1 features;
bow_stemmed: contain functions for vectorization with stemming and computing top m correlated words;
bow_lemmatized: contain functions for vectorization with lemmatization and computing top m correlated words; can be imported into main_stemmed for BoW 4 features;models: contains the simple models and a pipeline function that fits the model and returns accuracies;
preprocess_reviews: preprocess all reviews with lemmatization and export the results to a csv file for future use;
rnn: read lemmaztized reviews, creates word embedding matrices, and run LSTM; current hyperparameters in the file produced the plot in final report;
uni_and_bigrams: can perform kfold cross validation for linear models;
plot_corr: generate wordclouds presented in the report;
eda: initial exploratory data analysis;
split: split the original training set into training, validation, and testing;Final logistic regression: solver="sag", C=0.5Final linear SVM: C=0.3