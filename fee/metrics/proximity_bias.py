import numpy as np
from ..utils import get_g, get_pair_idb
from tqdm import tqdm
from collections import defaultdict

def bias_ratio(vals, l, thresh):
    return len(vals[vals>thresh]) / l

def get_nbs_i(E, word, n):
    return np.argsort(E.vecs.dot(E.v(word)))[-n:][::-1]

def _prox_bias(word, E, g=None, thresh=0.05, n=100):
    values = []
    neighbours_indices = get_nbs_i(E, word, n)
    for i, n_i in enumerate(neighbours_indices):
        values.append(float(get_pair_idb(word, E.vecs[n_i], g, E)))  
    return bias_ratio(np.array(values), n, thresh)

class ProxBias():
    def __init__(self, E, g=None, thresh=0.05, n=100):
        if g is None:
            g = get_g(E)        
        self.g = g
        self.E = E
        self.thresh = thresh
        self.n = n

    def compute(self, words):
        if not isinstance(words, list):
            words = [words]
        pb = np.mean([_prox_bias(w, self.E, 
                    self.g, self.thresh, self.n) for w in words])    
        return pb            