import numpy as np
from ..utils import get_g


class DirectBias():
    def __init__(self, E, c=1, g=None):
        if g is None:
            g = get_g(E)
        assert len(g) == E.dim   
        self.g = g
        self.E = E
        self.c = c

    def _direct_bias(self, vec):
       return np.power(np.abs(vec.dot(self.g)), self.c)     

    def compute(self, word_list):
        if not isinstance(word_list, list):
            word_list = [word_list]
        db = np.mean(
            [self._direct_bias(self.E.v(word)) for word in word_list]
        )
        return db
