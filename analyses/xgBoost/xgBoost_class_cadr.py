import csv
import json
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

import os
import re
import random
import operator
import nltk 
#nltk.download('stopwords')
from nltk import pos_tag
from nltk.corpus import stopwords
from collections import Counter

from xgboost import XGBClassifier

# fix for bug in venv on windows, probably starting from python 3.7.2: https://bugs.python.org/issue35797
# this manifests as "PermissionError: [WinError 5] Access is denied" errors.
# this is a workaround
in_virtual_env = sys.prefix != sys.base_prefix
if sys.platform == 'win32' and in_virtual_env and sys.version_info.major == 3 and sys.version_info.minor == 7:
    import _winapi
    sys.executable = _winapi.GetModuleFileName(0)

this_file_path = os.path.abspath(__file__)
this_file_path = '/Users/josehernandez/Documents/eScience/projects/cadrs/analyses/xgBoost/xgBoost_class_cadr.py'
project_root = os.path.split(os.path.split(os.path.split(this_file_path)[0])[0])[0]

sys.path.append(project_root)

import text_preprocess as tp

path_root = os.path.join(project_root, "data") + '/'
path_to_metadata = os.path.join(project_root, "metadata") + '/'
path_to_cadrs = path_root + 'cadrs/'


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

# we might want to get rid of duplication after standardization
dedup_fl = pd.concat([text,labels], axis = 1).drop_duplicates()
dedup_fl['subject_class'].value_counts()

text = dedup_fl['Name']
labels = dedup_fl['subject_class']

# begin algorithm prep
# use statify parameter to ensure balance between classes when data is split 
x_train, x_test, y_train, y_test = train_test_split(text, labels, stratify = labels ,test_size=0.2, random_state = 42)

#look at class sizes for training and test sets
y_train.value_counts()
y_test.value_counts()

sgd = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('xgb', XGBClassifier(
        eval_metric = 'auc',
        num_class = 7,
        nthread = 4,
        silent = 1,
        objective = 'multi:softprob',)),
               ])

parameters = {
    'vect__analyzer': ['word','char'],
    'vect__ngram_range': [(1,1), (1,2)],
    'tfidf__use_idf': (True, False),
    'xgb__num_boost_round': [100, 250, 500],
    'xgb__eta': [0.05, 0.1, 0.3],
    'xgb__max_depth': [6, 9, 12],
    'xgb__subsample': [0.9, 1.0],
    'xgb__colsample_bytree': [0.9, 1.0],
}

gs_clf = GridSearchCV(sgd, parameters, cv=5, iid=False, n_jobs=-1)
gs_clf.fit(x_train, y_train)

gs_clf.best_score_
for param_name in sorted(parameters.keys()):
    print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))

gs_clf.cv_results_
# check the test set for results 

test_pred = gs_clf.predict(x_test)
print('accuracy %s' % accuracy_score(test_pred, y_test))
print(classification_report(y_test, test_pred))
print(confusion_matrix(y_test, test_pred))

label_nm = labels.drop_duplicates()
conf_mat = confusion_matrix(y_test, test_pred, normalize='true')

from sklearn.metrics import plot_confusion_matrix

plot_confusion_matrix(gs_clf,x_test, y_test, xticks_rotation = 'vertical', normalize='true')

## Look at the expected outputs 
output = pd.DataFrame()
output['text'] = x_test
output['Expected Output'] = y_test
output['Predicted Output'] = test_pred
output.tail()
# USE CV ON ALL DATA 

######### 

len(text)

student_pred = gs_clf.predict(text)

len(student_pred)

pred_cols = pd.DataFrame(student_pred, columns = ['p_CADRS'])
pred_cols.head

combined_pred = crs_student.merge(pred_cols, left_index=True, right_index=True)
combined_pred.head
combined_pred.to_csv('/home/joseh/data/cnn_cadr_student_predictions_xgBoost_CV.csv', encoding='utf-8', index=False)

