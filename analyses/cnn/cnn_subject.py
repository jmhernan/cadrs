# try with pre-trained embeddings as a function of the model
import os
import re
import random
import sys
from pathlib import Path

import csv
import json
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import utils

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

this_file_path = os.path.abspath(__file__)
project_root = os.path.split(os.path.split(os.path.split(this_file_path)[0])[0])[0]

sys.path.append(project_root)

import text_preprocess as tp
import plotting as plot

path_root = os.path.join(project_root, "data") + '/'
path_to_metadata = os.path.join(project_root, "metadata") + '/'
path_to_cadrs = path_root + 'cadrs/'
path_to_db = project_root + '/output/'

# need to test the following (edge cases)
# run update on data
updated_cadrs = sorted(list(filter(lambda x: '.csv' in x, os.listdir(path_to_cadrs))))[-1]
crs_cat =  pd.read_csv(os.path.join(path_to_cadrs,updated_cadrs), delimiter = ',')
crs_abb = tp.get_metadata_dict(os.path.join(path_to_metadata, 'course_abb.json'))

# use the subject class as the label
text = crs_cat['Name'].apply(tp.clean_text)
text = tp.update_abb(text, json_abb=crs_abb).values.tolist()

label = crs_cat['subject_class'].values.tolist()

# max sequence of course title
num_words_row = [len(words.split()) for words in text]
max_seq_len = max(num_words_row)

# Tokenize
tokenizer = Tokenizer()
tokenizer.fit_on_texts(text)
word_index = tokenizer.word_index # word and their token # ordered by most frequent
print('Found %s unique tokens.' % len(word_index))
vocab_size = 589

sequences = tokenizer.texts_to_sequences(text)

# Padding
text_pad = pad_sequences(sequences, maxlen=max_seq_len+2)
text_pad.shape

# Handling the label
encoder = LabelEncoder()
label_arr = encoder.fit_transform(label)

# create dict of class to value for predictions
label_name_mapping = dict(zip(encoder.transform(encoder.classes_), encoder.classes_))
print(label_name_mapping)

num_classes = np.max(label_arr) + 1 #index at 0
label_pad = utils.to_categorical(label_arr, num_classes)

print('Shape of data tensor:', text_pad.shape)
print('Shape of label tensor:', label_pad.shape)

# Prep test and training 
x_train, x_val, y_train, y_val = train_test_split(text_pad, label_pad,
    test_size=0.2, random_state = 42)

# Build a model fun
VOCAB_SIZE = vocab_size
EMBEDDING_DIM = 100
NUM_CLASSES = num_classes

def cnn_text(n_filter, f_size, optimizer = 'adam'):
    model = tf.keras.Sequential([
    # Embedding layer
    tf.keras.layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM),
    # LSTM layer
    tf.keras.layers.Conv1D(filters=n_filter, kernel_size=f_size, activation='relu'),
    tf.keras.layers.MaxPool1D(pool_size=5),
    tf.keras.layers.Dense(128, activation='relu', input_shape=(9,)),
    # output layer
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])
    return model

model = cnn_text(n_filter=64, f_size=3)
######
from keras.layers import Input, Embedding, Dropout, Conv1D, GlobalMaxPooling1D, Dense, Concatenate, MaxPooling1D, Flatten
from keras.models import Model, load_model
from keras.layers import SpatialDropout1D

VOCAB_SIZE = vocab_size
EMBEDDING_DIM = 100
NUM_CLASSES = num_classes
EMBEDDING_MATRIX = np.zeros((VOCAB_SIZE, EMBEDDING_DIM))
EMBEDDING_MATRIX.shape

nb_filters = 128
filter_size_a = 3
drop_rate = 0.5
my_optimizer = 'adam'

my_input = Input(shape=(None,))
embedding = Embedding(input_dim=EMBEDDING_MATRIX.shape[0], input_length=max_seq_len,
    output_dim=EMBEDDING_DIM, trainable=True,)(my_input)
        
x = Conv1D(filters = nb_filters, kernel_size = filter_size_a,
    activation = 'relu',)(embedding)
x = SpatialDropout1D(drop_rate)(x)
x = MaxPooling1D(pool_size=5)(x)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
prob = Dense(NUM_CLASSES, activation = 'softmax',)(x)
model = Model(my_input, prob)

model.compile(loss='categorical_crossentropy', optimizer = my_optimizer,
    metrics = ['accuracy']) 
model.summary()

num_epochs = 50
history = model.fit(x_train, y_train, 
    epochs=num_epochs, validation_data=(x_val, y_val), 
    verbose=2)

plot.plot_graphs(history, "accuracy")

plot.plot_graphs(history, "loss")

####
my_input = Input(shape=(None,))
embedding = Embedding(input_dim=EMBEDDING_MATRIX.shape[0], input_length=max_seq_len,
    output_dim=EMBEDDING_DIM, trainable=True,)(my_input)

embedding_dropped = Dropout(drop_rate)(embedding)

# A branch
conv_a = Conv1D(filters = 128,
              kernel_size = 3,
              activation = 'relu',
              )(embedding_dropped)

pooled_conv_a = GlobalMaxPooling1D()(conv_a)

pooled_conv_dropped_a = Dropout(drop_rate)(pooled_conv_a)

# B branch
conv_b = Conv1D(filters = 128,
              kernel_size = 2,
              activation = 'relu',
              )(embedding_dropped)

pooled_conv_b = GlobalMaxPooling1D()(conv_b)

pooled_conv_dropped_b = Dropout(drop_rate)(pooled_conv_b)

# C branch
conv_c = Conv1D(filters = 128,
              kernel_size = 4,
              activation = 'relu',
              )(embedding_dropped)

pooled_conv_c = GlobalMaxPooling1D()(conv_c)

pooled_conv_dropped_c = Dropout(drop_rate)(pooled_conv_c)

## concatenate
concat = Concatenate()([pooled_conv_dropped_a,pooled_conv_dropped_b,pooled_conv_dropped_c])

concat_dropped = Dropout(drop_rate)(concat)

prob = Dense(NUM_CLASSES, activation = 'softmax',)(concat_dropped)
model = Model(my_input, prob)
model.summary()

model.compile(loss='categorical_crossentropy', optimizer = my_optimizer,
    metrics = ['accuracy']) 

num_epochs = 30
history = model.fit(x_train, y_train, 
    epochs=num_epochs, validation_data=(x_val, y_val), 
    verbose=2)

plot.plot_graphs(history, "accuracy")

plot.plot_graphs(history, "loss")

# Read sqlite query results into a pandas DataFrame
import sqlite3

db = path_to_db + 'ccerCadrDB.db'
db = "/Users/josehernandez/Documents/eScience/projects/cadrs/card_db.db"
con = sqlite3.connect(db)
crs_student = pd.read_sql_query("SELECT * FROM ghf_tukwila17", con)
crs_student.shape
con.close()

crs_student.shape
crs_student.columns

crs_student['CourseTitle'] = crs_student['CourseTitle'].fillna("")
crs_abb = tp.get_metadata_dict(os.path.join(path_to_metadata, 'course_abb.json'))

# use the subject class as the label
p_text = crs_student['CourseTitle'].apply(tp.clean_text)
p_text = tp.update_abb(p_text, json_abb=crs_abb).values.tolist()

# max sequence of course title
num_words_row = [len(words.split()) for words in p_text]
max_seq_len = max(num_words_row)

# Tokenize
tokenizer = Tokenizer()
tokenizer.fit_on_texts(p_text)

sequences = tokenizer.texts_to_sequences(p_text)

# Padding
p_data = pad_sequences(sequences, maxlen=max_seq_len)
p_data.shape

print('Shape of data tensor:', p_data.shape)

predictions = model.predict(p_data)

indexes = np.argmax(predictions, axis=1)
# 

crs_student["predictions"] = pd.DataFrame(indexes)
crs_student["pred_text"] = crs_student["predictions"].map(labels_index)  
crs_student["proc_text"] = student_pred

test = crs_student[["CourseTitle", "proc_text"]]