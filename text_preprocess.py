import re
import json
import pandas as pd

# handle json
def get_metadata_dict(metadata_file):
    metadata_handle = open(metadata_file)
    metadata = json.loads(metadata_handle.read())
    return metadata

# clean text 
def clean_text(text, json_abb=None):
    # regex conditions
    REPLACE_BY_SPACE_RE = re.compile(r'[/(){}\[\]\|@,;]')
    REM_USC = re.compile(r'(_)')
    SEP_CAPS = re.compile(r'(?<=[a-z])(?=[A-Z])')
    BAD_SYMBOLS_RE = re.compile(r'[\W]')
    #BAD_SYMBOLS_RE = re.compile(r'[^0-9a-z #+_]')
    REM_GRADE = re.compile(r'(th.(grade|GRADE))')
    REPLACE_NUM_RMN = re.compile(r"([0-9]+)|(i[xv]|v?i{0,3})$")    
    text = REM_USC.sub(' ', text)
    text = SEP_CAPS.sub(' ', text)
    text = str.lower(text)
    text = REPLACE_NUM_RMN.sub('', text)
    text = REPLACE_BY_SPACE_RE.sub(' ', text)
    text = BAD_SYMBOLS_RE.sub(' ', text)
    text = REM_GRADE.sub('', text) 
    text = ' '.join(word for word in text.split() if len(word)>1)
    if json_abb is not None:
        abb_cleanup = {r'\b{}\b'.format(k):v for k, v in json_abb.items()}
        text.replace(to_replace = d, regex=True)
        return text
    else:
        return text

def update_data(df, json_cadr=None, json_abb=None):
    if json_cadr is not None:
        df['cadr']= df['Name'].map(json_cadr).fillna(df['cadr'])
        return df['cadr']
    if json_abb is not None:
        abb_cleanup = {r'\b{}\b'.format(k):v for k, v in json_abb.items()}
        return abb_cleanup

def multi_class_df(df, cadr_methods):
    df.dropna(subset = ["content_area"], inplace=True)
    df.reset_index(inplace=True)
    df["subject_class"] = df["content_area"]
    df["subject_class"] = df["content_area"].replace(cadr_methods.get("cadr_categories"))
    df["subject_class"] = df["subject_class"].replace(cadr_methods.get("other_cadr"))
    df["subject_class"] = df["subject_class"].replace(cadr_methods.get("non_cadr"))
    my_query_index = df.query('cadr == 0').index
    df.iloc[my_query_index, 8] = "non_cadr"
    print(pd.crosstab(df.subject_class, df.cadr).sort_values(1, ascending=False))
    return df