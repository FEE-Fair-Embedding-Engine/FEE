import numpy as np
from ..utils import get_g, get_pair_idb, get_nbs
from tqdm import tqdm
from collections import defaultdict


def get_neighbors(N, word, k):
    """
        Args:
            N (dict[dict]): A dict of dict, storing the neighbours of
            each word with IDB values.
            word (str): The target word whose neighbours are to be
                        fecthed.
            k (int): top `k` neighbours.
        Returns:
            A list of top `k` neighbours for `word`.

    """
    return list(N[word].keys())[1:k+1]

def prox_bias(vals, l, thresh):
    """
        Returns:
            the ratio of total neighbours that have IDB above `thresh`.
    
    """
    return len(vals[vals>thresh]) / l

def get_ns_idb(E, word, g):
    """
        Args:
            E (WE class object): Word embeddings object.
            word (str): The word in consideration.
            g (np.array): Gender direction.
        Returns:
            A dictionary of top 100 neighbours of `word` and the 
            indirect bias between `word` and each neighbour.
    
    """
    tops = get_nbs(E, word, 101) 
    wv = E.v(word)
    d = dict(zip([E.words[v] for v in tops], [get_pair_idb(E.vecs[v], wv, g, E) for v in tops]))
    return d

def get_neighbors_idb_dict(E, words):
    """
    Args:
        Args:
            E (WE class object): Word embeddings object.
            word (str): The word in consideration.
    
    Returns:
        A dict of dicts, storing the neighbours of each word 
        with IDB values.

            # The key of larger dict resembles the source node, its value
            # is again a dict which has keys and values. These are
            # respetively the target node and the weight of an edge
            # that is conceptually formed between the two nodes 
            # (keys of two dicts).

    """
    g = get_g(E)
    neighbour_idb_dict = dict(zip([w for w in words], 
                            [get_ns_idb(E, w, g) for w in words]))
    return neighbour_idb_dict                        

def score(vals, weights):
    """
    Score the values and weights.
    
    Args:
        vals (dict): A dict of words and their corresponding proximity bias.
        weights (dict): The weights of an edge according to GIPE metric.
        
    Returns:
        The final computed GIPE score
    
    """
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
    """GIPE metric 

        Args:
            biased_words (list[str]): A list of string of words, on
                                      which GIPE will be computed.
            E_new (WE class object): Represents the new embedding object,
                                     which consists of the debiased embeddings.
            E_orig (WE class object): Represents the old/original embedding 
                                      object, which consists of the non-debiased
                                      embeddings.
        kwargs:
            g (np.array): Gender direction.
            thresh (float): The minimum indirect bias threshold, above which
                            the association between a word and its neighbour
                            is considered biased.
            n (int): The top `n` neighbours to be considered.
        
        Returns:
            The final computed GIPE score of a word embedding over the given 
            word lists, and the corresponding created BBN.
            
    """
    total = 0
    neighbours = {}
    incoming_edges = defaultdict(list)
    etas = {}
    N = get_neighbors_idb_dict(E_new, biased_words)

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
    """The GIPE metric class"""
    def __init__(self, E_new, E_orig=None, g=None, thresh=0.05, n=100):
        """
        GIPE

        Args:
            E_new (WE class object): Represents the new embedding object,
                                     which consists of the debiased embeddings.
            E_orig (WE class object): Represents the old/original embedding 
                                      object, which consists of the non-debiased
                                      embeddings.
        kwargs:
            g (np.array): Gender direction.
            thresh (float): The minimum indirect bias threshold, above which
                            the association between a word and its neighbour
                            is considered biased.
            n (int): The top `n` neighbours to be considered.
            
    """
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
        """
        Args:
            words (list[str]): A list of string of words, on
                                      which GIPE will be computed.

        Returns:
            The final computed GIPE score of a word embedding over the given 
            word lists, and the corresponding created BBN.
            
        """
        assert isinstance(words, list), "Argument words must be a list." 
        assert len(words)>1, "More than one word needed to compute the graph in GIPE." 

        return _gipe(words, self.E_new, self.E_orig, 
                        self.g, self.thresh, self.n)            