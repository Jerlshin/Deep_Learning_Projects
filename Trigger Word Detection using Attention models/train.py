import numpy as np
from pydub import AudioSegment
import random
import sys
import io
import os
import glob
import IPython
from td_utils import *

from keras.callbacks import ModelCheckpoint
from keras.models import Model, load_model, Sequential
from keras.layers import Dense, Activation, Dropout, Input, Masking, TimeDistributed, LSTM, Conv1D
from keras.layers import GRU, Bidirectional, BatchNormalization, Reshape
from keras.optimizers import Adam
from keras.models import model_from_json

Tx = 5511 # The number of time steps input to the model from the spectrogram
n_freq = 101 # Number of frequencies input to the model at each time step of the spectrogram

Ty = 1375 # The number of time steps in the output of our model

X = np.load("./XY_train/X0.npy")
Y = np.load("./XY_train/Y0.npy")

X = np.concatenate((X, np.load("./XY_train/X1.npy")), axis=0)
Y = np.concatenate((Y, np.load("./XY_train/Y1.npy")), axis=0)

Y = np.swapaxes(Y, 1, 2)

# Load preprocessed dev set examples
X_dev = np.load("./XY_dev/X_dev.npy")
Y_dev = np.load("./XY_dev/Y_dev.npy")


def modelf(input_shape):
    
    X_input = Input(shape = input_shape)
    
    X = Conv1D(196, kernel_size = 15, strides = 4)(X_input)  
    X = BatchNormalization()(X)                              
    X = Activation("relu")(X)                                
    X = Dropout(0.8)(X)                                      

    X = GRU(units = 128, return_sequences = True)(X)         
    X = Dropout(0.8)(X)                                     
    X = BatchNormalization()(X)                              
    
    X = GRU(units = 128, return_sequences = True)(X)         
    X = Dropout(0.8)(X)                                      
    X = BatchNormalization()(X)                              
    X = Dropout(0.8)(X)                               
    
    X = TimeDistributed(Dense(1, activation = "sigmoid"))(X) 


    model = Model(inputs = X_input, outputs = X)
    
    return model  

model = modelf(input_shape = (Tx, n_freq))
model.summary()

opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=0.01)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])


model.fit(X, Y, batch_size=20, epochs=100)

loss, acc = model.evaluate(X_dev, Y_dev)
print("Dev set accuracy = ", acc)


json_file = open('./models/model_new3.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights('./models/model_new3.h5')