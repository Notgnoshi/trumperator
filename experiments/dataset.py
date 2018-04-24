"""
This file handles the data wrangling from the compressed json dataset
to a numpy-friendly form.

Side note. Do not store zipped text files in a Git repository. Git can
significantly compress large amounts of text, but not if it's already compressed.
The Git repository I'm using for my dataset is *huge*, but most of that is because
Git has to save snapshots of binary files for its entire history. Further, these
binary files are updated every hour... This is a recipe for disaster.
"""

import json
import re
import string
import sys
from zipfile import ZipFile

import numpy as np


# The symbol set the preprocessed dataset will use
ALPHABET = frozenset(string.ascii_letters + string.punctuation + ' #@')


def remove_urls(vTEXT):
    """
        Remove URLS from a given string.
    """
    vTEXT = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', vTEXT, flags=re.MULTILINE)
    return vTEXT


def remove_mentions(vTEXT):
    """
        Remove @mentions from a given string.
    """
    vTEXT = re.sub(r'@\w+', '', vTEXT, flags=re.MULTILINE)
    return vTEXT


def preprocess(text, alphabet):
    """
        Preprocess text using the frozenset `alphabet` given. Will convert everything
        to lower case.
    """
    return filter(alphabet.__contains__, text.lower())


def load_dataset(files, verbose=True):
    """
        Parses the tweets out of the zipped dataset. Returns an iterable of preprocessed tweets.
    """
    dataset = []

    for archive in files:
        with ZipFile(archive) as z:
            # ZipFile's are collections of zipped files.
            for name in z.namelist():
                with z.open(name) as f:
                    if verbose:
                        print(f'reading {name}')
                    dataset += json.load(f)

    # Filter out the retweets
    dataset = filter(lambda t: t['is_retweet'] is False, dataset)
    # Get only the tweet text
    dataset = (t['text']for t in dataset)
    # Remove URLs from the tweets.
    dataset = (remove_urls(t) for t in dataset)
    # # Remove @mentions from tweets.
    # dataset = (remove_mentions(t) for t in dataset)
    # Preprocess each tweet, filtering out nonascii alphabetic
    dataset = [''.join(preprocess(t, ALPHABET)) for t in dataset]
    return dataset


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


def seq_data(dataset, seq_len, seq_step, verbose):
    """
        Turn the given list of tweets into a single corpus and vectorize
        the corpus by mapping each character to an integer.
    """

    corpus = ' '.join(dataset)
    characters = sorted(list(set(corpus)))
    num_chars = len(characters)

    # Character and integer conversion lookup tables
    c2i = dict((c, i) for i, c in enumerate(characters))
    i2c = dict((i, c) for i, c in enumerate(characters))

    sequences, next_chars = chunk_text(corpus, seq_len, seq_step)

    if verbose:
        print(f'corpus length: {len(corpus)}')
        print(f'num characters: {num_chars}')
        print(f'number of sequences: {len(sequences)}')

    return corpus, sequences, next_chars, c2i, i2c, num_chars


def vec_data(sequences, next_chars, seq_length, num_chars, char_to_indices, verbose):
    """
        Given the textual dataset, shuffle and vectorize it. The output this
        function returns can be *quite* large (4ish GB).
    """
    if verbose:
        print('Vectorizing sequences...')

    X = np.zeros((len(sequences), seq_length, num_chars), dtype=np.bool)
    y = np.zeros((len(sequences), num_chars), dtype=np.bool)

    for i, seq in enumerate(sequences):
        for t, char in enumerate(seq):
            # Each character is a one-hot encoded vector
            X[i, t, char_to_indices[char]] = 1
        y[i, char_to_indices[next_chars[i]]] = 1

    # An array of indices
    I = np.arange(X.shape[0])
    # A shuffled array of indices
    np.random.shuffle(I)

    if verbose:
        print(f'X shape: {X.shape}, Y shape: {y.shape}')
        print(f'X size: {sys.getsizeof(X) * 0.000001 :.3f} MB')
        print(f'y size: {sys.getsizeof(y) * 0.000001 :.3f} MB')

    # Return the shuffled arrays
    return X[I], y[I]
