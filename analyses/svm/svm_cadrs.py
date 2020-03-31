#!/usr/bin/env python
import csv
import numpy as np
import pandas as pd
import os
import re
from pathlib import Path

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
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV

# fix for bug in venv on windows, probably starting from python 3.7.2: https://bugs.python.org/issue35797
# this manifests as "PermissionError: [WinError 5] Access is denied" errors.
# this is a workaround
in_virtual_env = sys.prefix != sys.base_prefix
if sys.platform == 'win32' and in_virtual_env and sys.version_info.major == 3 and sys.version_info.minor == 7:
    import _winapi
    sys.executable = _winapi.GetModuleFileName(0)

this_file_path = os.path.abspath(__file__)
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

# look at class balance
crs_cat["cadr"].value_counts()
# handle json
# use the subject class as a "classifier"
text =  crs_cat['Name']
labels = crs_cat['cadr']

num_words = [len(words.split()) for words in text]
max(num_words)

text = text.apply(tp.clean_text)
text = tp.update_abb(text, json_abb=crs_abb)

# we might want to get rid of duplication after standardization
dedup_fl = pd.concat([text,labels], axis = 1).drop_duplicates()
dedup_fl['cadr'].value_counts()

text = dedup_fl['Name']
labels = dedup_fl['cadr']

# begin algorithm prep
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
gs_clf.fit(x_train, y_train)

gs_clf.best_score_
for param_name in sorted(parameters.keys()):
    print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))

gs_clf.cv_results_
# check the test set for results 
### with hyper parameters

# clf__alpha: 0.001
# tfidf__use_idf: True
# vect__analyzer: 'word'
# vect__ngram_range: (1, 2)

test_pred = gs_clf.predict(x_test)
print('accuracy %s' % accuracy_score(test_pred, y_test))
print(classification_report(y_test, test_pred))
print(confusion_matrix(y_test, test_pred))


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


