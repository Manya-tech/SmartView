from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.models import Sequential, load_model # Add 'load_model'
from joblib import dump, load # For reading the Tokenizer Pickle

KERAS_MODEL = "app/model.h5"
TOKENIZER_MODEL = "app/tokenizer.pkl"

# KERAS
SEQUENCE_LENGTH = 300

# SENTIMENT
POSITIVE = "Positive"
NEGATIVE = "Negative"
NEUTRAL = "Neutral"
SENTIMENT_THRESHOLDS = (0.4, 0.7)

# Load the model and the tokenizer to make predictions
model = load_model(KERAS_MODEL)
tokenizer = load(TOKENIZER_MODEL)

def decode_sentiment(score, include_neutral=False):
    if include_neutral:        
        label = NEUTRAL
        if score <= SENTIMENT_THRESHOLDS[0]:
            label = NEGATIVE
        elif score >= SENTIMENT_THRESHOLDS[1]:
            label = POSITIVE

        return label
    else:
        return NEGATIVE if score < 0.5 else POSITIVE
    
def predict_sentiment(text_list, include_neutral=False):
    sentiments=[]
    x_test = pad_sequences(tokenizer.texts_to_sequences(text_list), maxlen=SEQUENCE_LENGTH)
    # Predict
    scores = model.predict(x_test)
    for score in scores:

        label = decode_sentiment(score, include_neutral=include_neutral)
        sentiments.append(label)

    return sentiments
