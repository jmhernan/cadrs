"""
Functions specific to preprocess raw extract data from GoogleSheets CCER Course
 CADR codes.
"""
import re
import json
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# regex conditions for text cleanup
REPLACE_BY_SPACE_RE = re.compile(r'[/(){}\[\]\|@,;]')
REM_USC = re.compile(r'(_)')
SEP_CAPS = re.compile(r'(?<=[a-z])(?=[A-Z])')
BAD_SYMBOLS_RE = re.compile(r'[\W]')
REM_GRADE = re.compile(r'(th.(grade|GRADE))')
REM_CTE = re.compile(r'(\bCTE\b|\dCTE)')
REPLACE_NUM_RMN = re.compile(r"([0-9]+)|(i[xv]|v?i{0,3})$")

# handle json
def get_metadata_dict(metadata_file):
    metadata_handle = open(metadata_file)
    metadata = json.loads(metadata_handle.read())
    return metadata

# clean text 
def clean_text(text):    
    text = REM_USC.sub(' ', text)
    text = REM_CTE.sub('', text)
    text = SEP_CAPS.sub(' ', text)
    text = str.lower(text)
    text = REPLACE_NUM_RMN.sub('', text)
    text = REPLACE_BY_SPACE_RE.sub(' ', text)
    text = BAD_SYMBOLS_RE.sub(' ', text)
    text = REM_GRADE.sub('', text) 
    text = ' '.join(word for word in text.split() if len(word)>1)
    return text

def update_abb(text, json_abb):
    abb_cleanup = {r'\b{}\b'.format(k):v for k, v in json_abb.items()}
    abb_replace = text.replace(to_replace =abb_cleanup, regex=True)
    return abb_replace

def update_data(df, json_cadr=None, json_abb=None): #depricated use update abb cadrs coding happens in goolge docs 
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

def get_top_n_words(corpus, n=None):
    """
    List the top n words in a vocabulary according to occurrence in a text corpus.
    """
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]