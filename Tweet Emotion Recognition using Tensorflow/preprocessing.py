import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import nlp 
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


from utils import *


def extract_text_and_labels(data_df):
    if 'text' in data_df.columns and 'label' in data_df.columns:
        text_data = data_df['text'].tolist()
        label_data = data_df['label'].tolist()
        return text_data, label_data

def tokenizer(text):
    tokenizer = Tokenizer(num_words=10000, oov_token='<UNK>')
    tokenizer.fit_on_texts(text)
    return tokenizer, text



def get_sequences(tokernizer, tweets, max_len):
    sequences = tokernizer.texts_to_sequences(tweets)
    padded = pad_sequences(sequences, truncating='post', padding='post', maxlen=max_len)
    return padded