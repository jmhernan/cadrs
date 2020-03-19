# testing the pipeline with toy data + cadr assessments by subject 
import pandas as pd 
import os

this_file_path = os.path.abspath(__file__)
project_root = os.path.split(os.path.split(os.path.split(this_file_path)[0])[0])[0]

sys.path.append(project_root)
import text_preprocess as tp

pred_data = #test data HERE 

# First test complete 4 year HS students 
# with complete semesters thorough 12th grade
def cadr_test():
    # steps
    # 1. run sql scripts one for the complete set and another for the partial set 
    # 2. load data
    # 3. print report for overall and school % by subject 
    # 3. check numbers for categories against BERC manual codes
    #   a. overall %
    #   b. by school (if possible) or district 
def clean_text(text):
    # test this out
    REPLACE_BY_SPACE_RE = re.compile(r'[/(){}\[\]\|@,;]')
    BAD_SYMBOLS_RE = re.compile(r'[^0-9a-z #+_]')
    REM_GRADE = re.compile(r'\b[0-9]\w+')
    REPLACE_NUM_RMN = re.compile(r"([0-9]+)|(i[xv]|v?i{0,3})$")    
    text = text.lower()
    text = REM_GRADE.sub('', text)
    text = REPLACE_NUM_RMN.sub('', text)
    text = REPLACE_BY_SPACE_RE.sub(' ', text)
    text = BAD_SYMBOLS_RE.sub(' ', text) 
    text = ' '.join(word for word in text.split() if len(word)>1)
    print(text)

test_str =pd.Series([
    'BegELLReading',
    'BegELLWriting',
    'ENGINEER/DESIGN',
    'EngProfDev_2_S2',
    'Beg. Reading & Writing-ELL',
    'MarineSci_S1 9th Grade',
    'BandSymphonic_2',
    'BAND-CONCERT',
    'GraphicArt_1_S2',
    'ELLAdvSkills',
    'ELLBegLgArtsSup',
    'ELLIntLgArts Su',
    'EEL & JKI',
    'kjn#3',
    '10TH GRADE AVID 1-1',
    'Spanish IV',
    'Sci III',
    'ELD ENG 1 SP',
    'U.S. History',
    'Army Junior ROTC IV',
    'Algebra I—Part 2',
    'Portuguese IV',
    'Portuguese III'
])

test_str

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
    return text


def apply_abb(text, json_abb):
    abb_cleanup = {r'\b{}\b'.format(k):v for k, v in json_abb.items()}
    abb_replace = text.replace(to_replace =abb_cleanup, regex=True)
    return abb_replace

apply_abb(tt, json_abb=crs_abb)
