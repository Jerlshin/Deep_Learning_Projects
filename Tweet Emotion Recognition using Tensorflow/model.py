import tensorflow as tf
from keras.layers import Embedding
from keras.layers import Bidirectional, LSTM
from keras.layers import Dense


def model_0():
    max_len = 50
    model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(input_dim=10000, output_dim=16, input_length=max_len),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=20, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=20)),
    tf.keras.layers.Dense(6, activation='softmax')
    ])

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    return model

