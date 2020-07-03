# dowload google drive dataset
# function to work with the google api 
# you will need to enable your api and generate a key
# `GoogleAuth` will look for a "client_secrets.json" in the base directory
# You need to get the google drive file directory ID see analysis script for examples

import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
import os
from pathlib import Path

this_file_path = os.path.abspath(__file__)
# this_file_path = '/home/joseh/source/cadrs/'
project_root = os.path.split(this_file_path)[0]

path_root = os.path.join(project_root, "data") + '/'
path_to_metadata = os.path.join(project_root, "metadata") + '/'
path_to_cadrs = path_root + 'cadrs/'

import datetime

def _getToday():
	return datetime.date.today().strftime("%Y%m%d")

# use creds to create a client to interact with the Google Drive API
scope = ['https://spreadsheets.google.com/feeds','https://www.googleapis.com/auth/drive']
creds = ServiceAccountCredentials.from_json_keyfile_name(os.path.join(project_root, 'client_secret.json'), scope)
client = gspread.authorize(creds)

db_name = 'CCER_eScience_CADRS'
data = client.open(db_name)
data.worksheets()
data_sheet = 'training_06212020'

updated_data = data.worksheet(data_sheet)
list_of_hashes = updated_data.get_all_records()
headers = list_of_hashes.pop(0)
df = pd.DataFrame(list_of_hashes, columns=headers)
print(df.tail())

filename = "%s_%s.%s" % ("training_data_updated", _getToday() ,"csv")

df.to_csv(os.path.join(path_to_cadrs, filename), encoding='utf-8', index=False)



