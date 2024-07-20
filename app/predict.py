import numpy as np
from tensorflow import keras
from keras import preprocessing
from keras.models import load_model
from keras_preprocessing.sequence import pad_sequences
import pickle

VOCAB_SIZE = 10000
MAX_LEN = 250
MODEL_PATH = "app/sentiment_analysis_model.h5"


#Load the saved model
model = load_model(MODEL_PATH)

#Load the tokenizer
with open("app/tokenizer.pickle","rb") as handle:
    tokenizer = pickle.load(handle)


def encode_texts(text_list):
    encoded_texts = []
    for text in text_list:
        tokens = preprocessing.text.text_to_word_sequence(text)
        tokens = [tokenizer.word_index[word] if word in tokenizer.word_index else 0 for word in tokens]
        encoded_texts.append(tokens)
    return pad_sequences(encoded_texts, maxlen=MAX_LEN, padding='post', value=VOCAB_SIZE-1)

def predict_sentiment(text_list):
    encoded_inputs = encode_texts(text_list)
    predictions = np.argmax(model.predict(encoded_inputs), axis=-1)
    sentiments = []
    for prediction in predictions:
        if prediction == 0:
           sentiments.append("Negative")
        elif prediction == 1:
            sentiments.append("Neutral")
        else:
            sentiments.append("Positive")

    return sentiments
