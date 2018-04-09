#!/usr/bin/env python3
"""
This script takes a pretrained model and generates text.
"""

import random
import sys
from glob import glob

import numpy as np

from dataset import load_dataset, seq_data
from model import BASENAME, DIVERSITIES, GEN_LEN, SEQ_LEN, SEQ_STEP, load_model


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


def generate_sequence(model, seed, length, diversities,
                      seq_length, num_chars, char_to_indices, indices_to_char, verbose):
    """
        Given a model, generate a text given some seed.
    """
    g = []
    for diversity in diversities:
        generated = ''
        sequence = seed
        generated += seed

        if verbose:
            print('='*10)
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
        g.append(generated)
        if verbose:
            print(generated)
    return g


def save_generated(generated, filename):
    """
        Saves the generated text strings to the given file.
    """
    with open(f'{filename}.txt', 'a') as f:
        # Flatten the generated list because it's a list of samples, each with a different diversity
        for g in generated:
            f.write(g + '\n')


def generate(basename, model, corpus, c2i, i2c, num_chars, num_seeds, verbose):
    """
        Generate sequences of text from a given model, and save the generated
        text to a file.
    """
    # Generate text from seeds randomly taken from the corpus
    indices = [random.randint(0, len(corpus) - SEQ_LEN - 1) for _ in range(num_seeds)]
    seeds = [corpus[i: i + SEQ_LEN] for i in indices]
    generated = []
    for seed in seeds:
        docs = generate_sequence(model, seed, GEN_LEN, DIVERSITIES, SEQ_LEN, num_chars,
                                 c2i, i2c, verbose)
        save_generated(docs, basename)

    for seq in generated:
        print(seq)


def main(num_seeds, verbose):
    dataset = load_dataset(glob('../data/trump_tweet_data_archive/condensed_*.json.zip'), verbose)
    corpus, _, _, c2i, i2c, nc = seq_data(dataset, SEQ_LEN, SEQ_STEP, verbose)
    model = load_model(BASENAME, verbose)
    generate(BASENAME, model, corpus, c2i, i2c, nc, num_seeds, verbose)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        main(int(sys.argv[1]), verbose=True)
    else:
        main(10, verbose=True)
