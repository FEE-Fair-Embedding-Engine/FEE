import numpy as np
from ..utils import get_g, get_pair_idb, get_nbs
from tqdm import tqdm
from collections import defaultdict


def get_neighbors(N, word, k):
    return list(N[word].keys())[1:k+1]

def prox_bias(vals, l, thresh):
    return len(vals[vals>thresh]) / l

def get_ns_idb(E, word, g):
    tops = get_nbs(E, word, 101) 
    wv = E.v(word)
    d = dict(zip([E.words[v] for v in tops], [get_pair_idb(E.vecs[v], wv, g, E) for v in tops]))
    return d

def get_neighbors_idb_dict(words, E):
    g = get_g(E)
    neighbour_idb_dict = dict(zip([w for w in words], 
                            [get_ns_idb(E, w, g) for w in words]))
    return neighbour_idb_dict                        

def score(vals, weights):
    score = 0
    sum = 0
    for v in vals:
        try:
            score += weights[v] * vals[v]
            sum += weights[v]
        except:
            aux_w = 1  #By default, the weight is 1 (1 is the lowest possible weight, means lowest "penalty")
            score += vals[v] * aux_w
            sum += aux_w
    score /= sum
    return score

def _gipe(biased_words, E_new, E_orig=None, g=None, thresh=0.05, n=100):
    total = 0
    neighbours = {}
    incoming_edges = defaultdict(list)
    etas = {}
    N = get_neighbors_idb_dict(biased_words, E_new)

    # for word in tqdm(biased_words): #Creating BBN
    for word in biased_words: #Creating BBN
        try:
            neighbours[word] = get_neighbors(N, word, n) #Neighbours according to current embedding
            l = len(neighbours[word])
        except:
            print(f"{word} is weird.")
            continue
        values = []
        for i, element in enumerate(neighbours[word]):
            value = float(get_pair_idb(word, element, g, E_orig))  #Beta according to original (same in case of non-debiased) embedding
            values.append(value)
            incoming_edges[element].append(value)
        etas[word] = prox_bias(np.array(values), l, thresh)

    eps = np.finfo(float).eps
    weights = defaultdict(int)
    for key in incoming_edges:
        idbs = np.array(incoming_edges[key])
        weights[key] = 1 + (len(idbs[idbs>thresh])/(len(idbs) + eps))
    return score(etas, weights)



class GIPE():
    def __init__(self, E_new, E_orig=None, g=None, thresh=0.05, n=100):
        if not E_new.normalized:
            print("Normalizing...")
            E_new = E_new.normalize()

        if E_orig is not None and not E_orig.normalized:
            print("Normalizing...")
            E_orig = E_orig.normalize()        

        if E_orig is None:
            E_orig = E_new

        if g is None:
            g = get_g(E_new)        

        self.g = g
        self.E_new = E_new
        self.E_orig = E_orig
        self.thresh = thresh
        self.n = n


    def compute(self, words):
        return _gipe(words, self.E_new, self.E_orig, 
                        self.g, self.thresh, self.n)            