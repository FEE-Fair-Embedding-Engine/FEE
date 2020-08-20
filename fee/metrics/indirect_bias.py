import numpy as np
from ..utils import get_g


class IndirectBias():
    def __init__(self, E, g=None):
        if g is None:
            g = get_g(E)
        assert len(g) == E.dim   
        self.g = g
        self.E = E

    def _pair_idb(self, w, v, g):
        w_orth = w - (np.dot(w, g)) * g
        v_orth = v - (np.dot(v, g)) * g
        dots = np.dot(w, v)
        orth_dots = np.dot(w_orth, v_orth)
        idb = (dots - orth_dots / (np.linalg.norm(w_orth) * np.linalg.norm(v_orth))) / (dots )
        return idb        

    def fix(self, w, eps=1e-3):
        norm = np.linalg.norm(w)
        if np.abs(norm - 1) > eps:
            w /= norm
        return w    
    
    def compute(self, w, v):
        if isinstance(w, str): w = self.E.v(w)
        if isinstance(v, str): v = self.E.v(v)    
        w = self.fix(w) 
        v = self.fix(v) 
        return self._pair_idb(w, v, self.g)

