import csv
import numpy as np
import pandas as pd
import os
import re
import sys
from pathlib import Path

this_file_path = os.path.abspath(__file__)
# this_file_path = '/home/joseh/source/cadrs/analyses/svm/'
project_root = os.path.split(os.path.split(os.path.split(this_file_path)[0])[0])[0]
path_to_metadata = os.path.join(project_root, "metadata/")
sys.path.append(project_root)

import text_preprocess as tp

# Added non normalized courses to dict. Preprosess step needs to run.
test_courses = tp.get_metadata_dict(os.path.join(path_to_metadata, 'test_courses.json'))

# Need to add preprocessing step for cleaning 
test_courses = {tp.clean_text(key): value for key, value in test_courses.items()} 

# Abbreviations 
crs_abb = tp.get_metadata_dict(os.path.join(path_to_metadata, 'course_abb.json'))

clean_crs = tp.update_abb(pd.Series(list(test_courses.keys())), json_abb=crs_abb)
# Need to load saved model and run predictions
pred_courses = gs_clf.predict(clean_crs)

# append dict with predictions
for key, item in zip(test_courses, pred_courses):
    test_courses[key].extend([item])

# Convert to df to run validation
column_names = ["course", "list_values"]
val_df = pd.DataFrame(test_courses.items(), columns = column_names)
val_df[["type", "class", "prediction"]] = pd.DataFrame(val_df.list_values.tolist(),
    index=val_df.index)
val_df["updated_course_name"] = clean_crs

#run validation 
def is_valid(df, class_var, prediction_var):
    return(sum((df[class_var] == df[prediction_var]))/df.shape[0])

is_valid(val_df, class_var="class", prediction_var="prediction")
