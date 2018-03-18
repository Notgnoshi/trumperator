import json
import re
import string
from zipfile import ZipFile


def remove_urls(vTEXT):
    """
        Remove URLS from a given string.
    """
    vTEXT = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', vTEXT, flags=re.MULTILINE)
    return vTEXT


# Redefine what 'alphabet' means so we don't filter out spaces and hashtags.
ALPHABET = frozenset(string.ascii_lowercase + ' ' + '#')


def preprocess(text, use_ascii=True):
    """
        Preprocess text. Converts to lowercase and filters non-alphabetic characters.
        Defaults to defining alphabetic characters as ascii-alphabetic.

        Examples:
        >>> text = 'ABC.,#'
        >>> ''.join(preprocess(text))
        'abc'
        >>> text = 'ÈÆÖÉEAEOE,.%'
        >>> ''.join(preprocess(text, use_ascii=False))
        'èæöéeaeoe'
    """
    if use_ascii:
        return filter(ALPHABET.__contains__, text.lower())
    return filter(str.isalpha, text.lower())


def load_dataset(files):
    """
        Parses the tweets out of the zipped dataset. Returns a generator of preprocessed tweets.
    """
    dataset = []

    for archive in files:
        with ZipFile(archive) as z:
            # ZipFile's are collections of zipped files.
            for name in z.namelist():
                with z.open(name) as f:
                    print(f'reading {name}')
                    dataset += json.load(f)

    # Filter out the retweets
    dataset = filter(lambda x: x['is_retweet'] is False, dataset)
    # Get only the text
    dataset = (t['text']for t in dataset)
    # Remove URLs from the tweets.
    # TODO: Consider removing username mentions?
    dataset = (remove_urls(t) for t in dataset)
    # Preprocess each tweet, filtering out nonascii alphabetic
    dataset = (''.join(preprocess(t)) for t in dataset)

    return dataset
