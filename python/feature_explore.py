#!/usr/bin/env python
import csv
import numpy as np
import pandas as pd
import os
import re
import sys
from pathlib import Path
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import SGDClassifier

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

this_file_path = os.path.abspath(__file__)
#this_file_path = '/home/joseh/source/cadrs/analyses/svm/'
project_root = os.path.split(os.path.split(this_file_path)[0])[0]

sys.path.append(project_root)

import text_preprocess as tp

path_root = os.path.join(project_root, "data") + '/'
path_to_metadata = os.path.join(project_root, "metadata") + '/'
path_to_cadrs = path_root + 'cadrs/'
path_to_db = project_root + '/output/'

# need to test the following (edge cases)
updated_cadrs = sorted(list(filter(lambda x: '.csv' in x, os.listdir(path_to_cadrs))))[-1]
crs_cat =  pd.read_csv(os.path.join(path_to_cadrs,updated_cadrs), delimiter = ',')
crs_abb = tp.get_metadata_dict(os.path.join(path_to_metadata, 'course_abb.json'))

# look at class sizes 
crs_cat["subject_class"].value_counts()

# use the subject class as a "classifier"
text =  crs_cat['Name']
labels = crs_cat['subject_class']

num_words = [len(words.split()) for words in text]
max(num_words)

text = text.apply(tp.clean_text)
text = tp.update_abb(text, json_abb=crs_abb)

tp.get_top_n_words(text)

###### Pipeline
sgd_feat = Pipeline([('vect', CountVectorizer(ngram_range=(1, 2), analyzer='word')),
                ('tfidf', TfidfTransformer(use_idf= True))
               ])

#The full pipeline as a step in another pipeline with an estimator as the final step
model_pipeline = Pipeline( steps = [ ('feat_pipeline', sgd_feat),
                                     ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None)) 
                                  ])

# Analyse tfidf 
Xtr = sgd_feat.fit_transform(text)
vec = sgd_feat.named_steps['vect']
features = vec.get_feature_names()

def top_tfidf_feats(row, features, top_n=25):
    ''' Get top n tfidf values in row and return them with their corresponding feature names.'''
    topn_ids = np.argsort(row)[::-1][:top_n]
    top_feats = [(features[i], row[i]) for i in topn_ids]
    df = pd.DataFrame(top_feats)
    df.columns = ['feature', 'tfidf']
    return df


def top_feats_in_doc(Xtr, features, row_id, top_n=25):
    ''' Top tfidf features in specific document (matrix row) '''
    row = np.squeeze(Xtr[row_id].toarray())
    return top_tfidf_feats(row, features, top_n)

top_feats_in_doc(Xtr=Xtr, features=features,row_id=0,top_n=25)

def top_mean_feats(Xtr, features, grp_ids=None, min_tfidf=0.1, top_n=25):
    ''' Return the top n features that on average are most important amongst documents in rows
        indentified by indices in grp_ids. '''
    if grp_ids:
        D = Xtr[grp_ids].toarray()
    else:
        D = Xtr.toarray()

    D[D < min_tfidf] = 0
    tfidf_means = np.mean(D, axis=0)
    return top_tfidf_feats(tfidf_means, features, top_n)

top_mean_fts = top_mean_feats(Xtr=Xtr, features=features, grp_ids=None, min_tfidf=0.1, top_n=25)

def top_feats_by_class(Xtr, y, features, min_tfidf=0.1, top_n=25):
    ''' Return a list of dfs, where each df holds top_n features and their mean tfidf value
        calculated across documents with the same class label. '''
    dfs = []
    labels = np.unique(y)
    for label in labels:
        ids = np.where(y==label)
        feats_df = top_mean_feats(Xtr, features, ids, min_tfidf=min_tfidf, top_n=top_n)
        feats_df.label = label
        dfs.append(feats_df)
    return dfs

dfs = top_feats_by_class(Xtr=Xtr, y=labels, features=features, min_tfidf=0.1, top_n=25)

def plot_tfidf_classfeats_h(dfs, multiple=True):
    ''' Plot the data frames returned by the function plot_tfidf_classfeats(). '''
    if multiple == True:
        fig = plt.figure(figsize=(18, 10), facecolor="w")
        x = np.arange(len(dfs[0]))
        for i, df in enumerate(dfs):
            ax = fig.add_subplot(1, len(dfs), i+1)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.set_frame_on(False)
            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()
            ax.set_xlabel("Mean Tf-Idf Score", labelpad=16, fontsize=8)
            ax.set_title("label = " + str(df.label), fontsize=10)
            ax.ticklabel_format(axis='x', style='sci', scilimits=(-2,2))
            ax.barh(x, df.tfidf, align='center', color='#3F5D7D')
            ax.set_yticks(x)
            ax.set_ylim([-1, x[-1]+1])
            yticks = ax.set_yticklabels(df.feature)
            plt.subplots_adjust(bottom=0.09, right=0.97, left=0.15, top=0.95, wspace=0.52)
        plt.show()
    else:
        df = dfs
        fig = plt.figure(figsize=(18, 10), facecolor="w")
        x = np.arange(len(df))
        ax = fig.add_subplot()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_frame_on(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        ax.set_xlabel("Mean Tf-Idf Score", labelpad=16, fontsize=8)
        ax.set_title("Misclassification", fontsize=10)
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-2,2))
        ax.barh(x, df.tfidf, align='center', color='#3F5D7D')
        ax.set_yticks(x)
        ax.set_ylim([-1, x[-1]+1])
        yticks = ax.set_yticklabels(df.feature)
        plt.show()

plot_tfidf_classfeats_h(dfs)

# Work on this function also look at the other code for CV
model_pipeline.fit(x_train, y_train)
y_pred = model_pipeline.predict(x_test)
print('accuracy %s' % accuracy_score(y_pred, y_test))

def analyze_model(model=None, y=y, X=X, folds=10):
    ''' Run x-validation and return scores, averaged confusion matrix, and df with false positives and negatives '''

    y = y.values   # to numpy
    X = X.values

    # Manual x-validation to accumulate actual
    cv_skf = StratifiedKFold(n_splits=folds, shuffle=False, random_state=42)
    scores = []
    conf_mat = np.zeros((7, 7))      # Binary classification
    false_class = set()

    for train_i, val_i in cv_skf.split(X,y):
        X_train, X_val = X[train_i], X[val_i]
        y_train, y_val = y[train_i], y[val_i]

        print("Fitting fold...")
        model.fit(X_train, y_train)

        print ("Predicting fold...")
        y_plabs = np.squeeze(model.predict(X_val))  # Predicted class labels

        confusion = confusion_matrix(y_val, y_plabs)
        conf_mat += confusion
        score = accuracy_score(y_val, y_plabs) 
        scores.append(score)

        # Collect indices of false positive and negatives
        fc_i = np.where((y_plabs!=y_val))[0]
        false_class.update(val_i[fc_i])
        

        print("Fold score: ", score)
        print("Fold CM: \n", confusion)
        
    print("\nMean score:", np.mean(scores), np.std(scores)*2)
    conf_mat /= folds
    print("Mean CM: \n", conf_mat)
    return scores, conf_mat, {'misclass': sorted(false_class)}

test = analyze_model(model=model_pipeline, y=labels, X=text, folds=10)

X = text 
y = labels

y = y.values   # to numpy
X = X.values

cv_skf = StratifiedKFold(n_splits=10, shuffle=False, random_state=42)

# NEED SCORE TO GET MEAN 
conf_mat = np.zeros((7, 7))      # Binary classification
false_class = set()
scores = []

for train_i, val_i in cv_skf.split(X,y):
    X_train, X_val = X[train_i], X[val_i]
    y_train, y_val = y[train_i], y[val_i]

    print("Fitting fold...")
    model_pipeline.fit(X_train, y_train)

    print("Predicting fold...")
    y_plabs = np.squeeze(model_pipeline.predict(X_val))  # Predicted class labels

    confusion = confusion_matrix(y_val, y_plabs)
    conf_mat += confusion
    score = accuracy_score(y_val, y_plabs)
    scores.append(score)

    # Collect indices of false positive and negatives
    fc_i = np.where((y_plabs!=y_val))[0]
    false_class.update(val_i[fc_i])
    

    print("Fold score: ", score)
    print("Fold CM: \n", confusion)



print("\nMean score:", np.mean(scores), np.std(scores)*2)
conf_mat /= 10
print("Mean CM: \n", conf_mat)



len(false_class)

class_i = top_mean_feats(Xtr=Xtr, features=features, grp_ids=list(false_class), min_tfidf=0.1, top_n=25)

plot_tfidf_classfeats_h(class_i, multiple=False)

