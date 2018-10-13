import os
import itertools as it
import functools as ft
import functools
import operator as op

try:
    from functools import lru_cache
except:
    from repoze.lru import lru_cache

try:
    import cPickle as pickle
except:
    import pickle

import numpy as np
import csv
import gensim
import pandas as pd
import nltk
from nltk.corpus import wordnet as wn
from nltk.stem.porter import PorterStemmer
from sklearn.cross_validation import cross_val_score

from cross_validate import ClaimKFold


_max_ppdb_score = 10.0
_min_ppdb_score = -_max_ppdb_score


@lru_cache(maxsize=1)
def get_dataset(filename='url-versions-2015-06-14-clean.csv'):
    folder = os.path.join(_data_folder, 'emergent')
    return pd.DataFrame.from_csv(os.path.join(folder, filename))

_wnl = nltk.WordNetLemmatizer()

def normalize_word(w):
    return _wnl.lemmatize(w).lower()


def get_tokenized_lemmas(s):
    return [normalize_word(t) for t in nltk.word_tokenize(s)]


@lru_cache(maxsize=1)
def get_ppdb_data():
    with open(os.path.join(_pickled_data_folder, 'ppdb.pickle'), 'rb') as f:
        return pickle.load(f)


_stemmer = PorterStemmer()


@lru_cache(maxsize=100000)
def get_stem(w):
    return _stemmer.stem(w)


@lru_cache(maxsize=100000)
def compute_paraphrase_score(s, t):
    """Return numerical estimate of whether t is a paraphrase of s, up to
    stemming of s and t."""
    s_stem = get_stem(s)
    t_stem = get_stem(t)

    if s_stem == t_stem:
        return _max_ppdb_score

    # get PPDB paraphrases of s, and find matches to t, up to stemming
    s_paraphrases = set(get_ppdb_data().get(s, [])).union(get_ppdb_data().get(s_stem, []))
    matches = filter(lambda a, b, c: a == t or a == t_stem, s_paraphrases)
    if matches:
        return max(matches, key=lambda x, y, z: y)[1]
    return _min_ppdb_score

