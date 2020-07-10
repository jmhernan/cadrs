import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.svm import LinearSVC
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import chi2

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

from sklearn.linear_model import SGDClassifier

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

this_file_path = os.path.abspath(__file__)
#this_file_path = '/home/joseh/source/cadrs/analyses/svm/'
project_root = os.path.split(os.path.split(os.path.split(this_file_path)[0])[0])[0]

sys.path.append(project_root)

import text_preprocess as tp

path_root = os.path.join(project_root, "data") + '/'
path_to_metadata = os.path.join(project_root, "metadata") + '/'
path_to_cadrs = path_root + 'cadrs/'
path_to_db = project_root + '/output/'

# need to test the following (edge cases)
updated_cadrs = sorted(list(filter(lambda x: '.csv' in x, os.listdir(path_to_cadrs))))[-1]
updated_cadrs = "training_data_updated_test_C_20200624.csv"
crs_cat =  pd.read_csv(os.path.join(path_to_cadrs,updated_cadrs), delimiter = ',')
crs_abb = tp.get_metadata_dict(os.path.join(path_to_metadata, 'course_abb.json'))

# look at class sizes 
crs_cat["subject_class"].value_counts()

# use the subject class as a "classifier"
text =  crs_cat['Name']
labels = crs_cat['subject_class']

text = text.apply(tp.clean_text)
text = tp.update_abb(text, json_abb=crs_abb)

# we might want to get rid of duplication after standardization
dedup_fl = pd.concat([text,labels], axis = 1).drop_duplicates()
dedup_fl['subject_class'].value_counts()

# Create flags using regex
# AP
dedup_fl['ap'] = dedup_fl['Name'].str.contains(r"^.*?(advanced placement)").astype('uint8')

# ELL
dedup_fl['ell'] = dedup_fl['Name'].str.contains(r"^.*?(language learners)").astype('uint8')

fig = plt.figure(figsize=(8,6))
dedup_fl.groupby('subject_class').Name.count().plot.bar(ylim=0)

# begin the pipeline process

feat_pipe = Pipeline([
    ('union', FeatureUnion(
        transformer_list=[
            # pipeline for text
            ('text', Pipeline([
                ('vect', CountVectorizer(ngram_range=(1, 2), analyzer='word')),
                ('tfidf', TfidfTransformer(use_idf= True)),
            ])),
            # pipeline for metadata
            ('meta', Pipeline([
                ('flags', MetaTextTransformer()), 
                ('one_hot', OneHotEncoder(sparse=False)),
            ]))
        ]
    ))
])

X = dedup_fl['Name']
y =dedup_fl['subject_class']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40, stratify=y)

# Testing out encoder
one_hot = OneHotEncoder(sparse=False) 
one_hot.fit(X['ell'].to_numpy().reshape(-1,1))
test = one_hot.transform(X['ell'].to_numpy().reshape(-1,1))

#The full pipeline as a step in another pipeline with an estimator as the final step
model_pipeline = Pipeline( steps = [ ('feat_pipeline', feat_pipe),
                                     ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None)) 
                                  ])

#Can call fit on it just like any other pipeline
model_pipeline.fit(x_train, y_train)
y_pred = model_pipeline.predict(x_test)
print('accuracy %s' % accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

test = pd.Series(
    ['english language development',
    'advanced placement spanish',
    'english language development english']
)
model_pipeline.predict(test) # have the transformer create it by passing one name that can go through the right data type!!
##
feat_pipe_2 = Pipeline([
    ('union', FeatureUnion(
        transformer_list=[
            # pipeline for text
            ('text', Pipeline([
                ('vect', CountVectorizer(ngram_range=(1, 2), analyzer='word')),
                ('tfidf', TfidfTransformer(use_idf= True)),
            ]))
        ]
    ))
])

model_pipeline_2 = Pipeline( steps = [ ('feat_pipeline', feat_pipe_2),
                                     ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None)) 
                                  ])

#Can call fit on it just like any other pipeline
model_pipeline_2.fit(x_train, y_train)
y_pred = model_pipeline_2.predict(x_test)
print('accuracy %s' % accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

test = pd.Series(
    ['english language development',
    'advanced placement spanish',
    'english language development english']
)
model_pipeline_2.predict(test) # have the transformer create it by passing one name that can go through the right data type!!
