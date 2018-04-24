#!/usr/bin/env python3
"""
This script trains a recurrent LSTM network to generate tweets (hopefully)
sounding like those of President Trump.

After training the model, this script will save the model definition
and model weights to the specified file(s), as well as generate a text sample
when training is finished.
"""
from glob import glob

import matplotlib

# Hack to enable matplotlib to save a figure without an X server running (SSH sessions)
# https://stackoverflow.com/a/4706614
# Needs to run before `import matplotlib.pyplot as plt`
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

from dataset import load_dataset, seq_data, vec_data
from generate import generate
from model import (BASENAME, BATCH_SIZE, EPOCHS, PERCENT_VALIDATION,
                   SEQ_LEN, SEQ_STEP, build_model, save_model)

from textgenrnn.textgenrnn import textgenrnn

# Hack to keep Keras from allocating the whole damn gpu.
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = "0"

# Use just in time XLA compilation to hopefully eliminate the amount of data transfer to the GPU
config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
set_session(tf.Session(config=config))


def train_model(model, X, y, X_val=None, y_val=None, verbose=True):
    """
        Trains the given model. Will use validation data if given.
    """
    if X_val is not None and y_val is not None:
        if verbose:
            print(f'Training with {len(X_val)} validation samples')
        h = model.fit(X, y,
                      batch_size=BATCH_SIZE,
                      epochs=EPOCHS,
                      validation_data=(X_val, y_val))
    else:
        h = model.fit(X, y,
                      batch_size=BATCH_SIZE,
                      epochs=EPOCHS)
    return h


def plot_model_loss(filename, hist, verbose=True):
    """
        Given a Keras history object, save a plot of the training and
        validation losses to the given (extensionless) filename
    """
    if verbose:
        print(hist.history.keys())

    plt.plot(hist.history['loss'], label='training loss')

    if 'val_loss' in hist.history:
        plt.plot(hist.history['val_loss'], label='validation loss')
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.title('training loss')
    plt.legend()
    plt.savefig(f'{filename}.png')


def main(verbose):
    dataset = load_dataset(glob('../data/trump_tweet_data_archive/condensed_*.json.zip'), verbose)
    corpus, sequences, next_chars, c2i, i2c, nc = seq_data(dataset, SEQ_LEN, SEQ_STEP, verbose)

    if verbose:
        print(f'corpus length: {len(corpus)}')
        print(f'num characters: {nc}')
        print(f'number of sequences: {len(sequences)}')

    # The data is shuffled so the validation data isn't simply the latest 20% of tweets
    X, y = vec_data(sequences, next_chars, SEQ_LEN, nc, c2i, verbose)
    # Split off the last 20% as validation data for pretty graphs
    n = len(X)
    num_val = int(PERCENT_VALIDATION * n)
    X_val = X[n - num_val:]
    y_val = y[n - num_val:]

    X_train = X[:n - num_val]
    y_train = y[:n - num_val]

    if verbose:
        print(f'Number validation samples: {num_val}')

    model = build_model(SEQ_LEN, nc, verbose)
    history = train_model(model, X_train, y_train, X_val, y_val, verbose)
    plot_model_loss(BASENAME, history, verbose)
    # Save the trained model so we don't have to wait 25 hours to generate another 10 tweet sample
    save_model(model, BASENAME, verbose)
    # Generate sample tweets using 10 random seeds from the corpus.
    generate(BASENAME, model, corpus, c2i, i2c, nc, 10, verbose)

    # # Train the Textgenrnn model on our dataset. The README has information on how to take
    # # a trained model and generate tweets. (It's three lines of code)
    # gen = textgenrnn()
    # gen.train_new_model(
    #     dataset,
    #     # Label each tweet as a Trump tweet. Useful when combining multiple sources
    #     context_labels=['trump' for t in dataset],
    #     num_epochs=6,
    #     gen_epochs=1,
    #     batch_size=128,
    #     prop_keep=1.0,
    #     rnn_layers=2,
    #     rnn_size=128,
    #     rnn_bidirectional=False,
    #     max_length=40,
    #     dim_embeddings=100,
    #     word_level=False,
    # )
    # gen.generate(10, temperature=0.2)


if __name__ == '__main__':
    main(verbose=True)
