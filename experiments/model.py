#!/usr/bin/env python3
import random
import sys
from datetime import datetime
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.layers.core import Dense, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential, model_from_json

from dataset import load_dataset

# Hack to keep keras from allocating the whole damn gpu.
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = "0"
#session = tf.Session(config=config)
set_session(tf.Session(config=config))


def chunk_text(corpus, length, step):
    """
        Cut the given corpus into a series of offset sequences.
    """

    # A list of sequences
    sequences = []
    # For each sequence, record the next character in the corpus
    next_chars = []
    for i in range(0, len(corpus) - length, step):
        sequences.append(corpus[i:i + length])
        next_chars.append(corpus[i + length])
    return sequences, next_chars


def vectorize_data(corpus, sequences, next_chars, seq_length, num_chars, char_to_indices, verbose):
    """
        Given the textual dataset, vectorize it.
    """
    if verbose:
        print('Vectorizing sequences...')
    X = np.zeros((len(sequences), seq_length, num_chars), dtype=np.bool)
    y = np.zeros((len(sequences), num_chars), dtype=np.bool)
    for i, seq in enumerate(sequences):
        for t, char in enumerate(seq):
            X[i, t, char_to_indices[char]] = 1
        y[i, char_to_indices[next_chars[i]]] = 1

    return X, y


def build_model(seq_length, num_chars, verbose):
    """
        Builds an LSTM model for generating text. One layer should be enough.
    """
    if verbose:
        print('Building model...')
    model = Sequential()
    # TODO: Model size/depth?
    # TODO: LSTM options?
    model.add(LSTM(256, input_shape=(seq_length, num_chars)))
    # TODO: Regularization?
    # model.add(Dropout(0.2))
    model.add(Dense(num_chars, activation='softmax'))

    # TODO: Try different loss functions, optimizers, and metrics
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    return model


def sample_predictions(preds, temperature=1.0):
    """
        Helper function to sample an index from a probability array.
    """
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def train_model(model, X, y, X_val=None, y_val=None, verbose=True):
    """
        Trains the given model. Will use validation data if given.

        Will also save loss plots.
    """
    BATCH_SIZE = 128
    EPOCHS = 50

    if X_val and y_val:
        h = model.fit(X, y,
                      batch_size=BATCH_SIZE,
                      epochs=EPOCHS,
                      validation_data=(X_val, y_val))
    else:
        h = model.fit(X, y,
                      batch_size=BATCH_SIZE,
                      epochs=EPOCHS)

    if verbose:
        print(h.history.keys())

    plt.plot(h.history['loss'], label='training loss')

    if X_val and y_val:
        plt.plot(h.history['val_loss'], label='validation loss')
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.title('training loss')
    plt.legend()
    plt.savefig('training_loss.png')

    return h


def save_model(model, verbose, filename=None):
    """
        Saves the given model to disk, using a generated filename if one is not given.
    """
    if not filename:
        timestamp = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
        filename = f'model-{timestamp}'

    if verbose:
        print(f'Saving model definition to {filename}.json')
        print(f'saving model weights to {filename}.h5')

    with open(f'{filename}.json', 'w') as json:
        json.write(model.to_json())
    model.save_weights(f'{filename}.h5')


def load_model(base_filename, verbose):
    """
        Loads the given model from disk. The model consists of two files, the model definition,
        saved in json, and the model weights, saved in HDF5.
    """
    if verbose:
        print(f'Loading model definition from {base_filename}.json')
        print(f'Loading model weights from {base_filename}.h5')

    with open(f'{base_filename}.json') as json:
        model = model_from_json(json.read())
    model.load_weights(f'{base_filename}.h5')
    return model


def generate_sequence(model, corpus, seed, length, diversities,
                      seq_length, num_chars, char_to_indices, indices_to_char, verbose):
    """
        Given a model, generate a text given some seed.
    """
    # TODO: Does len(seed) == LEN?

    for diversity in diversities:
        generated = ''
        sequence = seed
        generated += seed

        if verbose:
            print(f'Generating with diversity: {diversity}')
            print(f'Generating with seed: "{seed}"')
            print(f'Generated:')

        for _ in range(length):
            X_pred = np.zeros((1, seq_length, num_chars))
            for t, c in enumerate(sequence):
                X_pred[0, t, char_to_indices[c]] = 1

            predictions = model.predict(X_pred, verbose=0)[0]
            next_index = sample_predictions(predictions, diversity)
            next_char = indices_to_char[next_index]
            generated += next_char
            # TODO: Don't use a list for this, use a queue
            sequence = sequence[1:] + next_char

        if verbose:
            print(generated)


def main(train, verbose):
    corpus = load_dataset(glob('../data/trump_tweet_data_archive/condensed_*.json.zip'), verbose)
    corpus = ' '.join(corpus)
    characters = sorted(list(set(corpus)))
    num_chars = len(characters)

    char_to_indices = dict((c, i) for i, c in enumerate(characters))
    indices_to_char = dict((i, c) for i, c in enumerate(characters))

    # The length of the sequences
    LEN = 60
    # How the far apart the sequences are spaced
    STEP = 3

    sequences, next_chars = chunk_text(corpus, LEN, STEP)

    if verbose:
        print(f'corpus length: {len(corpus)}')
        print(f'num characters: {num_chars}')
        print(f'number of sequences: {len(sequences)}')

    X, y = vectorize_data(corpus, sequences, next_chars, LEN, num_chars, char_to_indices, verbose)

    if verbose:
        print(f'X shape: {X.shape}, Y shape: {y.shape}')
        print(f'X size: {sys.getsizeof(X) * 0.000001 :.3f} MB')
        print(f'y size: {sys.getsizeof(y) * 0.000001 :.3f} MB')

    n = len(X)
    # Use 20% validation data
    num_val = int(0.2 * n)
    X_val = X[num_val:]
    y_val = y[num_val:]

    X_train = X[:num_val]
    y_train = y[:num_val]

    if verbose:
        print(f'Number validation samples: {num_val}')
        print(f'X_train shape: {X_train.shape}, Y_train shape: {y_train.shape}')
        print(f'X_train size: {sys.getsizeof(X_train) * 0.000001 :.3f} MB')
        print(f'y_train size: {sys.getsizeof(y_train) * 0.000001 :.3f} MB')
        print(f'X_val shape: {X_val.shape}, Y_val shape: {y_val.shape}')
        print(f'X_val size: {sys.getsizeof(X_val) * 0.000001 :.3f} MB')
        print(f'y_val size: {sys.getsizeof(y_val) * 0.000001 :.3f} MB')

    if train:
        model = build_model(LEN, num_chars, verbose)
        h = train_model(model, X_train, y_train, X_val=X_val, y_val=y_val, verbose=verbose)
        save_model(model, verbose, filename='latest')
    else:
        model = load_model('latest', verbose)

    # Generate text from seeds randomly taken from the corpus
    indices = [random.randint(0, len(corpus) - LEN - 1) for _ in range(10)]
    seeds = [corpus[i: i + LEN] for i in indices]
    generated = []
    for seed in seeds:
        generate_sequence(model, corpus, seed, 200, [0.05, 0.1, 0.2, 0.5], LEN, num_chars,
                          char_to_indices, indices_to_char, verbose)

    for seq in generated:
        print(seq)


if __name__ == '__main__':
    train = '--train' in sys.argv
    main(verbose=True, train=train)
