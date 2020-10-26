import numpy as np
from ..utils import get_g, get_pair_idb
from tqdm import tqdm
from collections import defaultdict

def _get_nbs_i(E, word, n):
    return np.argsort(E.vecs.dot(E.v(word)))[-n:][::-1]

def _pmb(word, E, g, n):
    values = []
    neighbours_indices = _get_nbs_i(E, word, n)
    male_neighbours = 0
    for i, n_i in enumerate(neighbours_indices):
        if E.vecs[n_i].dot(g) < 0: #males have negative direct bias 
            male_neighbours += 1
    return 100*male_neighbours/n

class PMN():
    """ The class that computes the Percentage of Male Neighbours (PMN)
        in the top n neighbours for a word.
    """
    def __init__(self, E, g=None, n=100):
        """
        Args:
            E (WE class object): Word embeddings object.
        kwargs:
            g (np.array): Gender direction.
            n (int): Top `n` neighbours according to the cosine similarity.
        """
        if g is None:
            g = get_g(E)        
        self.g = g
        self.E = E
        self.n = n

    def compute(self, words):
        """
        Args: 
            words (str or list[str]): A word or a list of worrds to
                                      compute the PMN for.
        Reutrn:
            The percentage of male neighbours. Note that the remaining
            percentage of neighbours can be considered to be female.
        """
        if not isinstance(words, list):
            words = [words]
        return np.mean([_pmb(w, self.E, 
                    self.g, self.n) for w in words]) 