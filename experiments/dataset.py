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

def remove_mentions(vTEXT):
    """
        Remove @mentions from a given string.
    """
    vTEXT = re.sub(r'@\w+', '', vTEXT, flags=re.MULTILINE)
    return vTEXT


ALPHABET = frozenset(string.ascii_lowercase + string.punctuation + ' #@')


def preprocess(text, alphabet):
    """
        Preprocess text using the frozenset `alphabet` given.
    """
    return filter(alphabet.__contains__, text.lower())

def load_dataset(files, verbose=True):
    """
        Parses the tweets out of the zipped dataset. Returns a generator of preprocessed tweets.
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
    dataset = filter(lambda x: x['is_retweet'] is False, dataset)
    # Get only the text
    dataset = (t['text']for t in dataset)
    # Remove URLs from the tweets.
    dataset = (remove_urls(t) for t in dataset)
    # Preprocess each tweet, filtering out nonascii alphabetic
    dataset = (''.join(preprocess(t, ALPHABET)) for t in dataset)

    return dataset
