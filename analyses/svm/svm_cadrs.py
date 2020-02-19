#!/usr/bin/env python
import csv
import json
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, cross_val_predict

from gensim.models.word2vec import Word2Vec

import os
import os.path
import re
import random
import operator
import sys
import nltk 
#nltk.download('stopwords')
from nltk import pos_tag
from nltk.corpus import stopwords
from collections import Counter
from bs4 import BeautifulSoup

from xgboost import XGBClassifier

# fix for bug in venv on windows, probably starting from python 3.7.2: https://bugs.python.org/issue35797
# this manifests as "PermissionError: [WinError 5] Access is denied" errors.
# this is a workaround
in_virtual_env = sys.prefix != sys.base_prefix
if sys.platform == 'win32' and in_virtual_env and sys.version_info.major == 3 and sys.version_info.minor == 7:
    import _winapi
    sys.executable = _winapi.GetModuleFileName(0)

this_file_path = os.path.abspath(__file__)
project_root = os.path.split(os.path.split(os.path.split(this_file_path)[0])[0])[0]
path_root = os.path.join(project_root, "data") + '/'
path_to_metadata = os.path.join(project_root, "metadata") + '/'
path_to_cadrs = path_root + 'cadrs/'
path_to_pretrained_wv = path_root



crs_cat =  pd.read_csv(os.path.join(path_to_cadrs,'cadrs_training_rsd.csv'), delimiter = ',')

# handle json
def get_metadata_dict(metadata_file):
    metadata_handle = open(metadata_file)
    metadata = json.loads(metadata_handle.read())
    return metadata
# load Json
crs_updates = get_metadata_dict(os.path.join(path_to_metadata, 'mn_crs_updates.json'))
crs_abb = get_metadata_dict(os.path.join(path_to_metadata, 'course_abb.json'))
# add regex
d = {r'\b{}\b'.format(k):v for k, v in crs_abb.items()}

# apply updates from Json
crs_cat['cadr'].describe()
crs_cat['cadr']= crs_cat['Name'].map(crs_updates).fillna(crs_cat['cadr'])
crs_cat['cadr'].describe()  

# Create lists of texts and labels 
text =  crs_cat['Name']
labels = crs_cat['cadr']

num_words = [len(words.split()) for words in text]
max(num_words)

# Clean up text
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
REM_GRADE = re.compile(r'\b[0-9]\w+')
REPLACE_NUM_RMN = re.compile(r"([0-9]+)|(i[xv]|v?i{0,3})$")


def clean_text(text):
    text = text.lower() # lowercase text
    text = REM_GRADE.sub('', text)
    text = REPLACE_NUM_RMN.sub('', text)
    text = REPLACE_BY_SPACE_RE.sub(' ', text)
    text = BAD_SYMBOLS_RE.sub(' ', text) 
    text = ' '.join(word for word in text.split() if len(word)>1)
    return text

text = text.apply(clean_text)
text = text.replace(to_replace = d, regex=True)

text.apply(lambda x: len(x.split(' '))).sum()
text[1:50]

# beggin algorithm prep
x_train, x_test, y_train, y_test = train_test_split(text, labels, test_size=0.2, random_state = 42)


sgd = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None)),
               ])

# cross_val_score(sgd, x_train, y_train, cv = 5)

# sgd.fit(x_train, y_train)

# y_pred = sgd.predict(x_test)

# print('accuracy %s' % accuracy_score(y_pred, y_test))
# print(classification_report(y_test, y_pred))
# print(confusion_matrix(y_test, y_pred))
#### GRD SEARCH CV for hyperparameter tunning 
parameters = {
    'vect__analyzer': ['word','char'],
    'vect__ngram_range': [(1,1), (1,2)],
    'tfidf__use_idf': (True, False),
    'clf__alpha': (1e-2, 1e-3),
}

gs_clf = GridSearchCV(sgd, parameters, cv=5, iid=False, n_jobs=-1)
gs_clf.fit(text, labels)

gs_clf.best_score_
for param_name in sorted(parameters.keys()):
    print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))

gs_clf.cv_results_
# check the test set for results 

test_pred = gs_clf.predict(x_test)

pred_cols = pd.DataFrame(test_pred, columns = ['p_CADRS'])


pred_cols.head

combined_pred = pred_cols.merge(x_test, left_index=True, right_index=True)
combined_pred = combined_pred.merge(y_test, left_index=True, right_index=True)
combined_pred.head




# Need to use saved model but using right after a fresh run for now
#file_nm = '/home/ubuntu/data/db_files/preprocess/course_2017_cohort_clean.csv'
file_nm = os.path.join(project_root, 'output/course_2017_cohort_clean.csv')
crs_student =  pd.read_csv(file_nm, delimiter = ',', dtype={'Description': str})
crs_student.shape
crs_student.columns

crs_student['state_spec_course']=crs_student['state_spec_course'].fillna("")

text_out =  crs_student['state_spec_course']
num_words_2 = [len(words.split()) for words in text_out]
max(num_words_2)

text = text_out.apply(clean_text)
text = text.replace(to_replace = d, regex=True)

text.apply(lambda x: len(x.split(' '))).sum()

text = text.astype(str).values.tolist()

len(text)

student_pred = gs_clf.predict(text)

len(student_pred)

pred_cols = pd.DataFrame(student_pred, columns = ['p_CADRS'])
pred_cols.head

combined_pred = crs_student.merge(pred_cols, left_index=True, right_index=True)
combined_pred.head
combined_pred.to_csv(os.path.join(path_root, 'svm_cadr_student_predictions_CV.csv'), encoding='utf-8', index=False)


