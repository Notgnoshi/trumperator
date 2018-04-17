"""
This file defines all of the model parameters to be
used by the rest of the workflow.
"""

from datetime import datetime
# from keras.layers import LSTM, Dense, BatchNormalization
from keras.models import Sequential, model_from_json

from textgenrnn.textgenrnn import textgenrnn

# Turn the corpus into sequences 60 characters long
SEQ_LEN = 60
# Make each sequence offset from each other by 3 characters
SEQ_STEP = 3
# Use 20% validation data
PERCENT_VALIDATION = 0.2
# List of diversities with which to sample the probability array
DIVERSITIES = [0.05, 0.1, 0.2, 0.5]
# Length of the generated sequences
GEN_LEN = 280
# Default is 32.
BATCH_SIZE = 128
# Number of epochs to train for. RMSprop is *much* faster than SGD and only needs 5-10 epochs
EPOCHS = 50
# Extensionless filename to save model configuration, weights, plots, and generated text to
BASENAME = 'models/64x6-rms-batchnorm-50'


def build_model(verbose):
    """
        Builds an LSTM model for generating text.
    """
    if verbose:
        print('Building model...')
    # model = Sequential()
    # model.add(LSTM(units=64, return_sequences=True, input_shape=(seq_length, num_chars), unit_forget_bias=True))
    # model.add(BatchNormalization())
    # model.add(LSTM(units=64, return_sequences=True, unit_forget_bias=True))
    # model.add(BatchNormalization())
    # model.add(LSTM(units=64, return_sequences=True, unit_forget_bias=True))
    # model.add(BatchNormalization())
    # model.add(LSTM(units=64, return_sequences=True, unit_forget_bias=True))
    # model.add(BatchNormalization())
    # model.add(LSTM(units=64, return_sequences=True, unit_forget_bias=True))
    # model.add(BatchNormalization())
    # model.add(LSTM(units=64, unit_forget_bias=True))
    # model.add(BatchNormalization())
    # model.add(Dense(num_chars, activation='softmax'))

    # model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    model = textgenrnn(name='trumperator')
    return model


def load_model(basename, verbose):
    """
        Loads the given model from disk. The model consists of two files, the model definition,
        saved in json, and the model weights, saved in HDF5.
    """
    if verbose:
        print(f'Loading model definition from {basename}.json')
        print(f'Loading model weights from {basename}.h5')

    with open(f'{basename}.json') as json:
        model = model_from_json(json.read())
    model.load_weights(f'{basename}.h5')
    return model


def save_model(model, basename=None, verbose=True):
    """
        Saves the given model to disk, using a generated filename if one is not given.
    """
    if not basename:
        timestamp = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
        basename = f'model-{timestamp}'

    if verbose:
        print(f'Saving model definition to {basename}.json')
        print(f'saving model weights to {basename}.h5')

    with open(f'{basename}.json', 'w') as json:
        json.write(model.to_json())
    model.save_weights(f'{basename}.h5')
