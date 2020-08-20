import numpy as np
from ..utils import get_g, get_pair_idb
from tqdm import tqdm
from collections import defaultdict

def get_nbs_i(E, word, n):
    return np.argsort(E.vecs.dot(E.v(word)))[-n:][::-1]

def _pmb(word, E, g, n):
    values = []
    neighbours_indices = get_nbs_i(E, word, n)
    male_neighbours = 0
    for i, n_i in enumerate(neighbours_indices):
        if E.vecs[n_i].dot(g) > 0: 
            male_neighbours += 1
    return 100*male_neighbours/n

class PMN():
    def __init__(self, E, g=None, n=100):
        if g is None:
            g = get_g(E)        
        self.g = g
        self.E = E
        self.n = n

    def compute(self, words):
        if not isinstance(words, list):
            words = [words]
        return np.mean([_pmb(w, self.E, 
                    self.g, self.n) for w in words]) 