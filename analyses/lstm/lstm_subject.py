# Try with pre-trained embeddings as a function of the model being built
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
text_pad = pad_sequences(sequences, maxlen=max_seq_len+1)
text_pad.shape

# Handling the label
encoder = LabelEncoder()
label_arr = encoder.fit_transform(label)

num_classes = np.max(label_arr) + 1 #index at 0
label_pad = utils.to_categorical(label_arr, num_classes)

print('Shape of data tensor:', text_pad.shape)
print('Shape of label tensor:', label_pad.shape)

# Prep test and training 
x_train, x_val, y_train, y_val = train_test_split(text_pad, label_pad,
    test_size=0.2, random_state = 42)

# Build a model fun
embedding_dim = 100
dropout = .25

model = tf.keras.Sequential([
    # Embedding layer
    tf.keras.layers.Embedding(vocab_size, embedding_dim),
    # LSTM layer
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    # Dense layer
    tf.keras.layers.Dense(embedding_dim, activation='relu'),
    # output layer
    tf.keras.layers.Dropout(dropout),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.summary()

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

num_epochs = 30
history = model.fit(x_train, y_train, 
    epochs=num_epochs, validation_data=(x_val, y_val), 
    verbose=2)

plot.plot_graphs(history, "accuracy")

plot.plot_graphs(history, "loss")

def pad_to_size(vec, size):
  zeros = [0] * (size - len(vec))
  vec.extend(zeros)
  return vec

def sample_predict(sample_pred_text, pad):
  encoded_sample_pred_text = encoder.encode(sample_pred_text)

  if pad:
    encoded_sample_pred_text = pad_to_size(encoded_sample_pred_text, 64)
  encoded_sample_pred_text = tf.cast(encoded_sample_pred_text, tf.float32)
  predictions = model.predict(tf.expand_dims(encoded_sample_pred_text, 0))

  return (predictions)

# create function for predictions of a lot of text 
# plus single smaller batches
Xnew = ['political science']

tokenizer_test = Tokenizer()
tokenizer_test.fit_on_texts(Xnew)
test_sequences = tokenizer.texts_to_sequences(Xnew)

test_data = pad_sequences(test_sequences, maxlen=max_seq_len+1)
test_data
predictions = model.predict(test_data)

###
label_sorted_ls = sorted(crs_cat['subject_class'])
label_names = np.unique(label_sorted_ls)


label_values = list(range(0,7))
#label_values = list(range(0,12))

labels_index = dict(zip(label_names, label_values))

score_dict = {label_index: predictions[0][idx] for idx, label_index in enumerate(labels_index)}

score_dict
# Add return max feature