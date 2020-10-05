import numpy as np
from ..utils import get_g, get_pair_idb
from tqdm import tqdm
from collections import defaultdict

def _bias_ratio(vals, l, thresh):
    return len(vals[vals>thresh]) / l

def _get_nbs_i(E, word, n):
    return np.argsort(E.vecs.dot(E.v(word)))[-n:][::-1]

def _prox_bias(word, E, g=None, thresh=0.05, n=100):
    values = []
    neighbours_indices = _get_nbs_i(E, word, n)
    for i, n_i in enumerate(neighbours_indices):
        values.append(float(get_pair_idb(word, E.vecs[n_i], g, E)))  
    return _bias_ratio(np.array(values), n, thresh)

class ProxBias():
    def __init__(self, E, g=None, thresh=0.05, n=100):
        """
        Args: 
            E (WE class object): Word embeddings object.

        kwargs:
            g (np.array): Gender direction.
            thresh (float): The minimum indirect bias threshold, above 
                            which the association between a word and its
                            neighbour is considered biased.
            n (int): Top `n` neighbours according to the cosine similarity.
        """
        if g is None:
            g = get_g(E)        
        self.g = g
        self.E = E
        self.thresh = thresh
        self.n = n

    def compute(self, words):
        """
        Args:
            words (str): A word or a list of worrds to compute the 
                         ProxBias for.
        Returns:
            The average proximity bias for the given list of `words`.
            Proximity bias is in simple terms the ratio of biased nieghbours
            according to indirect bias with respect to a word. 

        """
        if not isinstance(words, list):
            words = [words]
        pb = np.mean([_prox_bias(w, self.E, 
                    self.g, self.thresh, self.n) for w in words])    
        return pb            