import numpy as np
from ..utils import get_g


class IndirectBias():
    """ The class for computing indirect bias between a pair of words.
    """ 
    def __init__(self, E, g=None):
        """
        Args: 
            E (WE class object): Word embeddings object.
        kwargs:
            g (np.array): Gender direction.
        """
        if g is None:
            g = get_g(E)
        assert len(g) == E.dim   
        self.g = g
        self.E = E

    def _pair_idb(self, w, v, g):
        """
        Args: 
           w (np.array): The first word vector.  
           v (np.array): The second word vector.
           g (np.array): Gender direction.
        
        Returns:
            idb (float): The indirect bias between the embeddings of
                         `w` and `v`.
        """
        
        w_orth = w - (np.dot(w, g)) * g
        v_orth = v - (np.dot(v, g)) * g
        dots = np.dot(w, v)
        orth_dots = np.dot(w_orth, v_orth)
        idb = (dots - orth_dots / (np.linalg.norm(w_orth) * np.linalg.norm(v_orth))) / (dots )
        return idb        

    def fix(self, w, eps=1e-3):
        """
        Args:
            w (np.array): word vector
        kwargs:
            eps (float): threshold. If the difference between the norm
                         of `w` and 1, is greater than `eps`. Then 
                         normalize `w`.
        
        Returns:
            The normalized `w`.
            
        """
        norm = np.linalg.norm(w)
        if np.abs(norm - 1) > eps:
            w /= norm
        return w    
    
    def compute(self, w, v):
        """
        Args: 
           w (str): One of a pair of words.  
           v (str): The other word from the pair.
        
        Returns:
            The indirect bias between the embeddings of `w` and `v`.
        """
        if isinstance(w, str): w = self.E.v(w)
        if isinstance(v, str): v = self.E.v(v)    
        w = self.fix(w) 
        v = self.fix(v) 
        return self._pair_idb(w, v, self.g)

