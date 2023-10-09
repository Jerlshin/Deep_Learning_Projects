from preprocessing import extract_text_and_labels, tokenizer, get_sequences
from model import model_0
from utils import import_data, show_confusion_matrix, show_history, get_index

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import nlp 
import pandas as pd

from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.callbacks import ProgbarLogger


train_path = './dataset/training.csv'
valid_path = './dataset/validation.csv'

train = pd.read_csv(train_path)
valid = pd.read_csv(valid_path)

train_text, train_labels = extract_text_and_labels(train)
valid_text, valid_labels = extract_text_and_labels(valid)

tokenizer_train, train_tok = tokenizer(train_text)
tokenizer_valid, valid_tok = tokenizer(valid_text)

max_len = 50 # max sequence length

padded_train_seq = get_sequences(tokenizer_train, train_tok, max_len)
padded_valid_seq = get_sequences(tokenizer_valid, valid_tok, max_len)

class_to_index, index_to_class = get_index()

train_labels = np.array(train_labels)
valid_labels = np.array(valid_labels)



model = model_0()

checkpoint_callback = ModelCheckpoint(
    filepath='weights.h5',
    monitor='val_accuracy',
    save_best_only=True,
    save_weights_only=True,
    mode='max',
    verbose=1 # Display messages when saving weights
)

hypothesis = model.fit(
    x=padded_train_seq,
    y=train_labels,
    validation_data=(padded_valid_seq, valid_labels),
    epochs=100,
    callbacks=[
        EarlyStopping(monitor='val_accuracy', patience=2),
        checkpoint_callback
    ],
    verbose=1 
)

model.save('Model.h5')

