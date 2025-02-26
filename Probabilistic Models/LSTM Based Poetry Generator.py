# This script is adopted from, for learning purposes.
# https://www.geeksforgeeks.org/lstm-based-poetry-generation-using-nlp-in-python/
import string
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.utils as ku

from wordcloud import WordCloud
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers

# Configure GPU usage
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Configure GPU memory growth to prevent allocating all GPU memory at once
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPUs configured successfully")
    except RuntimeError as e:
        print("Error configuring GPUs:", e)

# Verify GPU availability
print("Num GPUs Available: ", len(gpus))

# Force GPU usage for the entire script
with tf.device('/GPU:0'):
    data = open('../Datasets/nlp.txt', encoding="utf8").read()
    wordcloud = WordCloud(max_font_size=200, max_words=90, background_color="black").generate(data)

    plt.figure(figsize=(19,10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.savefig("Wordclound.png")


    corpus = data.lower().split("\n")
    corpus_cleaned = [re.sub(f"[{string.punctuation}]", "", line) for line in corpus]

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(corpus_cleaned)
    total_words = len(tokenizer.word_index)
    print("Total words: ", total_words)

    input_sequences = []
    for line in corpus_cleaned:
        token_list = tokenizer.texts_to_sequences([line])[0]

        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)

    max_sequence_len = max([len(x) for x in input_sequences])
    input_sequences = np.array(pad_sequences(input_sequences,
                                             maxlen=max_sequence_len,
                                             padding='pre'))
    predictors, label = input_sequences[:, :-1], input_sequences[:, -1]
    label = ku.to_categorical(label, num_classes=total_words+1)

    # Model creation with GPU-specific configuration
    model = Sequential()
    model.add(Embedding(input_dim=total_words + 1, output_dim=100, input_shape=(max_sequence_len - 1,)))
    model.add(Bidirectional(LSTM(150, return_sequences=True)))
    model.add(Dropout(0.2))
    model.add(LSTM(100))
    model.add(Dense((total_words + 1) // 2, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(total_words + 1, activation='softmax'))

    # Use mixed precision training for better GPU performance
    tf.keras.mixed_precision.set_global_policy('mixed_float16')

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())

    # Training with GPU
    history = model.fit(predictors, label, epochs=1, verbose=1)

    # Generating the text using trained model
    seed_text = "The world"
    next_words = 25
    output_text = ""

    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences(
            [token_list], maxlen=max_sequence_len-1,
            padding='pre')
        predicted = np.argmax(model.predict(token_list, verbose=0), axis=-1)

        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break

        seed_text += " " + output_word

    print(seed_text)