import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import numpy as np
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from utils import get_index

from keras.preprocessing.text import Tokenizer

# Load the trained model
model = load_model('./model/Model.h5')

# Create a function to preprocess text input and make predictions
def predict_emotion():
    class_to_index, index_to_class = get_index()

    user_input = text_entry.get()  # Get text input from the user
    if user_input:
        # Tokenize and preprocess the input text
        tokenizer = Tokenizer(num_words=10000, oov_token='<UNK>')
        tokenizer.fit_on_texts(user_input)

        max_len = 50

        user_seq = tokenizer.texts_to_sequences([user_input])
        padded_user_seq = pad_sequences(user_seq, truncating='post', padding='post', maxlen=max_len)
        
        # Make a prediction using the loaded model
        prediction = model.predict(padded_user_seq)
        
        # Get the emotion label with the highest probability
        predicted_label = index_to_class[np.argmax(prediction)]
        
        # Display the result in a message box
        result_message = f"Predicted Emotion: {predicted_label}"
        messagebox.showinfo("Emotion Prediction Result", result_message)
    else:
        messagebox.showwarning("Empty Input", "Please enter text for prediction.")

# Create the main GUI window
root = tk.Tk()
root.title("Emotion Prediction")

# Create a label and text entry field
label = ttk.Label(root, text="Enter Text:")
label.pack(pady=10)
text_entry = ttk.Entry(root, width=40)
text_entry.pack(pady=5)

# Create a button to make predictions
predict_button = ttk.Button(root, text="Predict Emotion", command=predict_emotion)
predict_button.pack(pady=10)

# Start the GUI application
root.mainloop()
