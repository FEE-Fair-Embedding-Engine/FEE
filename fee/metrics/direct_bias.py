import numpy as np
from ..utils import get_g


class DirectBias():
    def __init__(self, E, c=1, g=None):
        """Direct bias calculation
        Args:
            E (WE class object): Word embeddings object
        Kwargs:
            c (float): strictness factor
            g (np.array): gender direction

        """
        if g is None:
            g = get_g(E)
        assert len(g) == E.dim   
        self.g = g
        self.E = E
        self.c = c

    def _direct_bias(self, vec):
        """Direct bias computation

        Args:
            vec (np.array): numpy array to calculate direct bias for
        """
        return np.power(np.abs(vec.dot(self.g)), self.c)     

    def compute(self, word_list):
        """Compute direct bias

        Args:
            word_list (list): list of words to compute bias for. 
        Returns:
            The direct bias of each word in the `word_list`.
        """
        if not isinstance(word_list, list):
            word_list = [word_list]
        db = np.mean(
            [self._direct_bias(self.E.v(word)) for word in word_list]
        )
        return db
