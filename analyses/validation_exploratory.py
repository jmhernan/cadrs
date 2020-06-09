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
df_cadr = pd.read_sql_query("SELECT * from cadr_tuk_val", con)
df_cadr.shape

query_txt = '''SELECT 
            ResearchID,ReportSchoolYear,DistrictName,SchoolName,TermEndDate,
            Term,GradeLevelWhenCourseTaken,CourseID,CourseTitle,
            p_CADRS,CreditsEarned,StateCourseCode,
            StateCourseName,ContentAreaCode,ContentAreaName,dSchoolYear 
            from cadr_pred_tuk'''


df_course = pd.read_sql_query(query_txt, con)
df_course.shape

con.close()

df_cadr.columns
df_course.columns



