import os
from functools import lru_cache

import pickle

import nltk
from nltk.stem.porter import PorterStemmer

_max_ppdb_score = 10.0
_min_ppdb_score = -_max_ppdb_score

_data_folder = os.path.join(os.path.dirname(__file__), '..', 'saved_data')

_wnl = nltk.WordNetLemmatizer()

def normalize_word(w):
    return _wnl.lemmatize(w).lower()


def get_tokenized_lemmas(s):
    return [normalize_word(t) for t in nltk.word_tokenize(s)]


@lru_cache(maxsize=1)
def get_ppdb_data():
    with open(os.path.join(_data_folder, 'ppdb.pickle'), 'rb') as f:
        return pickle.load(f)


_stemmer = PorterStemmer()


@lru_cache(maxsize=100000)
def get_stem(w):
    return _stemmer.stem(w)


@lru_cache(maxsize=100000)
def compute_paraphrase_score(s, t):
    """Return numerical estimate of whether t is a paraphrase of s, up to
        stemming of s and t.
    """
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

