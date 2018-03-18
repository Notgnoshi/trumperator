#!/usr/bin/env python3

import glob
import random
import sys

import numpy as np
from dataset import load_dataset

# Hack to keep keras from allocating the whole damn gpu.
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = "0"
#session = tf.Session(config=config)
set_session(tf.Session(config=config))

from keras.callbacks import LambdaCallback
from keras.layers.core import Dense, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential

'''
    Example script to generate haiku Text.
    It is recommended to run this script on GPU, as recurrent
    networks are quite computationally intensive.
    If you try this script on new data, make sure your corpus
    has at least ~100k characters. ~1M is better.
'''

text = load_dataset(glob.glob('../data/trump_tweet_data_archive/condensed_*.json.zip'))
# turn the list of tweets into one big corpus
text = ' '.join(text)
print(f'corpus length: {len(text)}')

chars = sorted(list(set(text)))
print(f'total chars: {len(chars)}')
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of maxlen characters
maxlen = 120
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print(f'sequences: {len(sentences)}')

print('Vectorizing input...')
X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

print(f'X shape: {X.shape}, Y shape: {y.shape}')
print(f'X size: {sys.getsizeof(X) * 0.000001 :.3f} MB')
print(f'y size: {sys.getsizeof(y) * 0.000001 :.3f} MB')

# build the model: single LSTM
print('Building model...')
model = Sequential()
model.add(LSTM(512, input_shape=(maxlen, len(chars))))
model.add(Dropout(0.2))
model.add(Dense(len(chars), activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')


def sample(preds, temperature=1.0):
    """helper function to sample an index from a probability array"""
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def on_epoch_end(epoch, logs):
    """Function invoked at end of each epoch. Prints generated text."""
    print(f'----- Generating text after Epoch: {epoch}')

    start_index = random.randint(0, len(text) - maxlen - 1)
    for diversity in [0.1, 0.2, 0.5, 1.0, 1.2]:
        print(f'\n\n----- diversity: {diversity}')

        generated = ''
        sentence = text[start_index: start_index + maxlen]
        generated += sentence
        print(f'----- Generating with seed: "{sentence}"')
        print(f'----- Generated:')
        sys.stdout.write(generated)

        # Generate N characters
        N = 400
        for _ in range(N):
            X_pred = np.zeros((1, maxlen, len(chars)))
            for t, c in enumerate(sentence):
                X_pred[0, t, char_indices[c]] = 1.

            preds = model.predict(X_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()
    print()

print_callback = LambdaCallback(on_epoch_end=on_epoch_end)

print('Training model...')
h = model.fit(X, y,
              batch_size=128,
              epochs=60,
              callbacks=[print_callback])
