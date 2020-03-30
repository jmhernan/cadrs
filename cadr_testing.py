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

text = test_str.apply(tp.clean_text)
text = tp.update_abb(text, json_abb=crs_abb)

