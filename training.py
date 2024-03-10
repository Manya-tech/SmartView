import os
import numpy as np
import pandas as pd
from tensorflow import keras
import tensorflow as tf
from keras_preprocessing.sequence import pad_sequences
from keras.models import Sequential, load_model
from keras.layers import Embedding, Dense, GlobalAveragePooling1D
from keras.preprocessing.text import Tokenizer
import pickle
# import warnings
# warnings.filterwarnings('ignore')



#Parameters

VOCAB_SIZE = 10000
MAX_LEN = 250
EMBEDDING_DIM = 16
MODEL_PATH = "sentiment_analysis_model.h5"

file_path = "YouTube-Sentiment-Analysis-App\data.csv"
df = pd.read_csv(file_path, encoding = 'ISO-8859-1')
df_shuffled = df.sample(frac=1).reset_index(drop=True)


texts = []
labels = []

print(df_shuffled.head())

for index, row in df_shuffled.iterrows():
   
    texts.append(row.iloc[-1])
    label = row.iloc[0]
    labels.append(0 if label == 0 else 1 if label == 2 else 2)

print("done")

texts = np.array(texts)
labels = np.array(labels)

#Tokenize the sequences
tokenizer = Tokenizer(num_words=VOCAB_SIZE)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

#Padding the sequences
padded_sequences = pad_sequences(sequences, maxlen = MAX_LEN, value=VOCAB_SIZE-1, padding='pre')
print(padded_sequences[0])

#Save the tokenizer to a file
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Split data into training and test sets 
train_data = padded_sequences[:-7000]
test_data = padded_sequences[-3000:]
train_labels = labels[:-7000]
test_labels = labels[-3000:]


# Check if saved model exists
if os.path.exists(MODEL_PATH):
    print("Loading saved model...")
    model = load_model(MODEL_PATH)
else:
    print("Training a new model...")
    
    # Create a MirroredStrategy.
    strategy = tf.distribute.MirroredStrategy()
    print("Number of devices: {}".format(strategy.num_replicas_in_sync))
    with strategy.scope():
    # Define the model
        model = Sequential([
        Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_LEN),
        GlobalAveragePooling1D(),
        Dense(16, activation='relu'),
        Dense(3, activation='softmax')  # 3 classes: negative, neutral, positive
    ])

        # Compile the model
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(train_data, train_labels, epochs=10, batch_size=32, validation_split=0.2)

    # Save the trained model
    model.save(MODEL_PATH)

    # Evaluate on test data
loss, accuracy = model.evaluate(test_data, test_labels)
print(f"Test accuracy: {accuracy * 100:.2f}%")

# Interactive loop for predictions
def encode_text(text):
    tokens = tf.keras.preprocessing.text.text_to_word_sequence(text)
    tokens = [tokenizer.word_index[word] if word in tokenizer.word_index else 0 for word in tokens]
    return pad_sequences([tokens], maxlen=MAX_LEN, padding='post', value=VOCAB_SIZE-1)

while True:
    user_input = input("Enter a sentence for sentiment analysis (or 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break
    
    encoded_input = encode_text(user_input)
    prediction = np.argmax(model.predict(encoded_input))

    if prediction == 0:
        print("Sentiment: Negative")
    elif prediction == 1:
        print("Sentiment: Neutral")
    else:
        print("Sentiment: Positive")