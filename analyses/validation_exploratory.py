#!/usr/bin/env python
# Read sqlite query results into a pandas DataFrame
import csv
import numpy as np
import pandas as pd
import os
import re
import sys
from pathlib import Path
import sqlite3

this_file_path = os.path.abspath(__file__)
project_root = os.path.split(os.path.split(this_file_path)[0])[0]

sys.path.append(project_root)

import text_preprocess as tp
 
path_root = os.path.join(project_root, "data") + '/'
path_to_metadata = os.path.join(project_root, "metadata") + '/'
path_to_cadrs = path_root + 'cadrs/'
path_to_db = project_root + '/' 

db = path_to_db + 'card_db.db'
con = sqlite3.connect(db)
df_cadr = pd.read_sql_query("SELECT * from Tukwila_test_agg_robust", con)
df_cadr.shape

query_txt = '''SELECT 
            ResearchID,ReportSchoolYear,DistrictName,SchoolName,TermEndDate,
            Term,GradeLevelWhenCourseTaken,CourseID,CourseTitle,
            p_CADRS,CreditsEarned,StateCourseCode,
            StateCourseName,ContentAreaCode,ContentAreaName,dSchoolYear 
            FROM cadr_pred_tuk'''


df_course = pd.read_sql_query(query_txt, con)
df_course.shape

con.close()

df_cadr.columns
df_course.columns

# df_cadr_sub = df_cadr[(df_cadr['CompleteHSRecords']== 1)]

df_cadr.to_csv(os.path.join(path_root, 'prediction_validation_log_06162020.csv'), encoding='utf-8', index=False)

# Output validation 

out_val =  pd.read_csv(os.path.join(path_root,'svm_cadr_output_val_06162020.csv'), delimiter = ',')

# Explore universe of course names in courses from 2016-present 
db = path_to_db + 'card_db.db'
con = sqlite3.connect(db)
cursor = con.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
print(cursor.fetchall())

query_txt = '''SELECT CourseTitle, ContentAreaName
            FROM hsCourses
            WHERE ReportSchoolYear > 2015
            '''

df_course = pd.read_sql_query(query_txt, con)
df_course.shape

con.close()
# run clean up procedure [FIX THIS!!]
dedup_df = df_course.drop_duplicates()

titles = dedup_df['CourseTitle'].apply(tp.clean_text)
titles = tp.update_abb(titles, json_abb=crs_abb)

titles = titles.drop_duplicates().reset_index()

titles = titles['CourseTitle'].to_list()
titles = [x for x in titles if x]

top_words = tp.get_top_n_words(titles)

# we need to see course titles with abbreviations
test = [0] * len(titles)
for index, title in enumerate(titles):
    test[index] = list(map(len, title.split()))

# change to select course titles that have words with less than 3 characters after clean up...

t = [
    'chs compar literature',
    'chs literature',
    'ocean meteor',
    'biology honors',
    'graphic art',
    'school annual',
    'modern world history honors',
    'digital design',
    'science ta'
]

for index, t in enumerate(t):  ``
    tn = list(map(len, t.split()))
    if min(tn) <= 3:
        print(index)
        
def abb_index(n,titles):
    indices = []
    for index, titles in enumerate(titles):
        tn = list(map(len, titles.split()))
        if min(tn) <= n:
            indices.append(index)
    return indices

test = abb_index(n=3,titles=titles)
len(test)
t_courses = [titles[i] for i in test] # we want to include the subject area if avalable 