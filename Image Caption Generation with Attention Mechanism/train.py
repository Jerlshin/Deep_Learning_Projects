import time 
import warnings

import pickle
import h5py
import os
from tqdm.notebook import tqdm

import random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
from tensorflow import keras

from keras.callbacks import TensorBoard
from keras.utils import Sequence
from keras.utils import to_categorical
from keras.models import Sequential, Model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
from keras.preprocessing.sequence import pad_sequences
from keras.layers import (
    Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D,
    Activation, Dropout,
    GRU, Bidirectional, LSTM,
    Add, add,
    AdditiveAttention,
    Attention,
    Concatenate, concatenate,
    Embedding,
    LayerNormalization,
    Dense,
    Reshape,
)
from sklearn.model_selection import train_test_split, KFold

from keras.layers.experimental.preprocessing import StringLookup
from keras.layers.experimental.preprocessing import TextVectorization

from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications import DenseNet201
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.utils import plot_model

from textwrap import wrap

from tensorflow.python.client import device_lib
print(tf.version.VERSION)
print(device_lib.list_local_devices())

plt.rcParams['font.size'] = 12
sns.set_style("dark")
warnings.filterwarnings('ignore')

DATA_DIR = '/home/jerlshin/Documents/My_Work/__DATASET__/flickr8k/'

IMAGE_DIR = os.path.join(DATA_DIR, 'images/')
CAPTIONS_DIR = os.path.join(DATA_DIR, 'captions.txt')

'''CONFIGURATION'''
class Param:    
    VOCAB_SIZE = 8485 # calculated with tokenizer 
    ATTENTION_DIM = 512
    WORD_EMBEDDDING = 128
    IMG_HEIGHT = 299
    IMG_WIDTH = 299
    IMG_CHANNELS = 3
    FEATURES_SHAPE = (8, 8, 1536) 
    
    BUFFER_SIZE = 1000
    BATCH_SIZE = 16  # reducing the batch size for latptop compatability
    MAX_CAPTION_LEN = 64

data = pd.read_csv(CAPTIONS_DIR)

'''PREPROCESSING'''
def text_preprocessig(data):
    data['caption'] = data['caption'].apply(lambda x : x.lower())
    data['caption'] = data['caption'].apply(lambda x : x.replace("[^A-Za-z]", ""))
    data['caption'] = data['caption'].apply(lambda x : x.replace("/s+", ""))
    data['caption'] = data['caption'].apply(lambda x : " ".join(
        [word for word in x.split() if len(word) > 1]
    ))
    data['caption'] = "SOS " + data['caption'] + " EOS"
    return data

data = text_preprocessig(data)
captions = data['caption'].tolist()

'''TOKENIZER'''
tokenizer = Tokenizer()
tokenizer.fit_on_texts(captions)
vocab_size = len(tokenizer.word_index) + 1
max_length = max(len(caption.split()) for caption in captions)

images = data['image'].unique().tolist()
nimages = len(images)

split_index = round(0.85*nimages)
train_images = images[:split_index]
val_images = images[split_index:]

train = data[data['image'].isin(train_images)]
test = data[data['image'].isin(val_images)]

train.reset_index(inplace=True,drop=True)
test.reset_index(inplace=True,drop=True)


'''BUILDING THE DATASET'''
# load all the paths of the images
image_file_paths_load = [os.path.join(IMAGE_DIR, image) for image in data['image']]

def preprocess_captions(captions, tokenizer, max_length):
    sequences = tokenizer.texts_to_sequences(captions)
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')
    return padded_sequences

captions_processed = preprocess_captions(data['caption'], tokenizer, Param.MAX_CAPTION_LEN)

dataset = tf.data.Dataset.from_tensor_slices((image_file_paths_load, captions_processed))

def load_and_preprocess_image(image_path, caption):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=Param.IMG_CHANNELS)
    img = tf.image.resize(img, (Param.IMG_HEIGHT, Param.IMG_WIDTH))
    img = img/255.0
    return img, caption


def load_and_preprocess_image_wrapper(image_path, caption):
    img, caption = load_and_preprocess_image(image_path, caption)
    return img, caption


BASE_MODEL = tf.keras.applications.InceptionResNetV2(
    weights='imagenet', include_top=False,
    input_shape=(Param.IMG_HEIGHT, Param.IMG_WIDTH, Param.IMG_CHANNELS)
)

BASE_MODEL.trainable = False

def extract_image_features(image):
    img = tf.expand_dims(image, axis=0)  # Add batch dimension
    image_features = BASE_MODEL(img)  # (batch_size, 8, 8, 1536)
    image_features = tf.reshape(tensor=image_features, shape=(-1, image_features.shape[3]))
    return image_features  # (64, 1536)


dataset = dataset.map(load_and_preprocess_image_wrapper)

dataset = dataset.shuffle(Param.BUFFER_SIZE).batch(Param.BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)

'''BUILDING THE MODEL'''


BASE_MODEL.trainable = False

# Image input
image_input = Input(shape=(Param.IMG_HEIGHT, Param.IMG_WIDTH, Param.IMG_CHANNELS), name='image_input')

# Word input
word_input = Input(shape=(Param.MAX_CAPTION_LEN,), name='word_input')

# Feature extraction from images using pre-trained InceptionResNetV2
image_features = BASE_MODEL(image_input)
x = Reshape(
    target_shape=(Param.FEATURES_SHAPE[0] * Param.FEATURES_SHAPE[1], Param.FEATURES_SHAPE[2])
    )(image_features)

# Encoder
encoder_output = Dense(Param.ATTENTION_DIM, activation='relu')(x)

encoder_model = Model(inputs=image_input, 
                      outputs=encoder_output, name='encoder_model')

# Word embedding
word_embedding = Embedding(input_dim=Param.VOCAB_SIZE, output_dim=Param.ATTENTION_DIM)(word_input)

# LSTM layer
lstm_layer = LSTM(Param.ATTENTION_DIM, return_sequences=True, return_state=True)

# LSTM layer
lstm_output, lstm_state, _ = lstm_layer(word_embedding)

# Attention mechanism
context_vector = Attention()([lstm_output, encoder_output])
addition = Add()([lstm_output, context_vector])

layer_norm = LayerNormalization(axis=-1)
layer_norm_out = layer_norm(addition)

# Output layer
decoder_output_dense = Dense(Param.VOCAB_SIZE)
decoder_output = decoder_output_dense(layer_norm_out)

# Decoder
decoder_model = Model(inputs=[word_input, encoder_output], 
                      outputs=decoder_output, name='decoder_model')

# Generator model
Generator_Model = Model(inputs=[word_input, image_input], 
                        outputs=decoder_output, name='generator_model')


Generator_Model.summary()


'''TRAIN AND VALIDATION SET'''

total_samples = len(dataset)

split = 0.85
train_samples = int(split * total_samples)
val_samples = total_samples - train_samples

train_dataset = dataset.take(train_samples)
val_dataset = dataset.skip(train_samples)

print("Number of samples in training dataset:", len(train_dataset) * Param.BATCH_SIZE)
print("Number of samples in validation dataset:", len(val_dataset) * Param.BATCH_SIZE)


train_dataset = train_dataset.map(lambda image, caption: (image, caption))
val_dataset = val_dataset.map(lambda image, caption: (image, caption))


'''=====TRAINING====='''

epochs = 10
steps_per_epochs = len(train_dataset)
validation_steps = len(val_dataset)


checkpoint_path = "checkpoints/model_{epoch:02d}.h5"

'''Model Checkpoints'''

callbacks_list = [
    ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        save_best_only=False,
        save_freq='epoch'
    ),
    EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2, patience=2,
        min_lr=1e-6
    ),
    TensorBoard(
        log_dir="logs",
        histogram_freq=1
    )
]


optimizer = Adam()

'''Loss Function'''
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask
    return tf.reduce_mean(loss)

summary_writer = tf.summary.create_file_writer("logs")

train_loss_results = []
val_loss_results = []

print("==========TRAINING STARTED==========")

for epoch in tqdm(range(epochs), desc="Epochs", ncols=400):
    total_loss = 0
    num_batches = 0
    
    for batch in tqdm(train_dataset, desc=f"Batch: {num_batches + 1}", ncols=500):
        images, captions = batch # 32 set of images and captions
        
        with tf.GradientTape() as tape:
            predictions = Generator_Model([captions, images], training = True)
            mask = tf.math.logical_not(tf.math.equal(captions, 0))
            loss = loss_function(captions, predictions)
            masked_loss = tf.math.reduce_sum(loss * tf.cast(mask, dtype=loss.dtype)) / tf.math.reduce_sum(tf.cast(mask, dtype=loss.dtype))
                
        gradients = tape.gradient(masked_loss, Generator_Model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, Generator_Model.trainable_variables))
        
        total_loss += masked_loss
        num_batches += 1
    
    average_train_loss = total_loss / steps_per_epochs
    train_loss_results.append(average_train_loss.numpy())
    
    val_total_loss = 0
    val_num_batches = 0
    
    for val_batch in val_dataset:
        val_images, val_captions = val_batch
        val_predictions = Generator_Model([val_images, val_captions], training = False)
        val_loss = loss_function(val_captions, val_predictions)
        
        val_mask = tf.math.logical_not(tf.math.equal(val_captions, 0))
        val_masked_loss = tf.math.reduce_sum(val_loss * tf.cast(val_mask, dtype=val_loss.dtype)) / tf.math.reduce_sum(tf.cast(val_mask, dtype=val_loss.dtype))
        val_total_loss += val_masked_loss
        
        val_num_batches += 1

    average_val_loss = val_total_loss / validation_steps
    val_loss_results.append(average_val_loss.numpy())
    
    
    # Log training and validation loss to TensorBoard
    with summary_writer.as_default():
        tf.summary.scalar('train_loss', average_train_loss, step=epoch)
        tf.summary.scalar('val_loss', average_val_loss, step=epoch)
    
    print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {average_train_loss:.4f}, Validation Loss: {average_val_loss:.4f}")
    
    
    for callback in callbacks_list:
        callback.on_epoch_end(epoch, logs={'train_loss': average_train_loss, 'val_loss': average_val_loss})
        print("Saving....\n")
    
summary_writer.close()