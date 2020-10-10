import numpy as np
from tqdm import tqdm
from copy import deepcopy

import torch.nn.functional as F
import torch.nn as nn
import torch

from ..utils import get_g, get_pair_idb, get_nbs

def _get_N_info(E, words, sel_max=1, desel_max=100):
    N = get_neighbors_idb_dict(words, E)
    thresh, N_info = 0.05, {} 
    for w in words:
        sel, desel = [], []
        try:
            for key in N[w]:
                if N[w][key] >= thresh: desel.append(key)
                else: sel.append(key)
        except:
            print(f"Problem for word: {w}")
        N_info[w] = dict(zip(['selected', 'deselected'], [sel[:sel_max], desel[:desel_max]]))  
    return N_info

def get_ns_idb(word, N):
    """Get indirect bias and neighbours

    Args:
        word (str): word to get neighbours and pair idb
        N (dict): neighbourhood-idb dictionary

    """
    return N[word]

def calc_ns_idb(E, word, g):
    """Calculate neighbourhood dictionary for one word with indirect bias
    pair information.

    Args:
        word (str): word to compute neighbours and pair idb
        E (WE class object): Word embeddings object

    """
    tops = get_nbs(E, word, 101) 
    wv = E.v(word)
    d = dict(zip([E.words[v] for v in tops], [get_pair_idb(E.vecs[v], wv, g, E) for v in tops]))
    return d

def get_neighbors_idb_dict(words, E):
    """create neighbourhood dictionary for each word with indirect bias
    pair information.

    Args:
        words (list): list of words to compute neighbours and pair idb
        E (WE class object): Word embeddings object

    """
    g = get_g(E)
    neighbour_idb_dict = dict(zip([w for w in words], 
                            [calc_ns_idb(E, w, g) for w in words]))
    return neighbour_idb_dict       

def init_vector(word, E):
    """initialize vector for optimization
    
    Args:
        word (str): word to debias
        E (WE class object): Word embeddings object

    """
    v = deepcopy(E.v(word)) 
    return torch.FloatTensor(v)

def torch_cosine_similarity(X, vectors):
    """torch tensor cosine similarity for groups

    Args:
        X (torch.Tensor): torch tensor 1D
        vectors (torch.Tensor): torch tensor 2D

    """
    return torch.matmul(vectors, X) / (vectors.norm(dim=1) * X.norm(dim=0))

def ran_objective(X, sel, desel, g, ws):
    """objective function for RAN

    Args:
        X (torch.Tensor): original word vector
        sel (torch.Tensor): selection tensors for attraction
        X (torch.Tensor): deselection tensors for repulsion
        g (torch.Tensor): gender direction tensor
        ws (list): list of objective weights

    """
    w1, w2, w3 = ws
    A = torch.abs(torch_cosine_similarity(X, sel) - 1).mean(dim=0)/2
    if not isinstance(desel, bool):
        R = torch.abs(torch_cosine_similarity(X, desel)).mean(dim=0)
    else:
        R = 0 #nothing to repel
    N = torch.abs(X.dot(g)).mean(dim=0)    
    J = w1*R + w2*A + w2*N
    return J

#CPU
class RANOpt(nn.Module):
    """RAN objective optimization class.
    """
    def __init__(self, E, word, X, N, g, ws=[0.33, 0.33, 0.33], ripa=False):
        """ 
        
        Args:
            E (WE class object): Word embeddings object.
            word (np.array): the original vector to debias
            N (dict): neighbourhood dictionary
            g (np.array): Gender Direction, if None, it is computed again.
            ws (list): weights for RAN objective
            ripa (bool): use RIPA based neutralization or not

        """
        super(RANOpt, self).__init__()
        sel_max = 1
        desel_max = 100
        self.sel = N[word]['selected'][:sel_max]
        self.desel = N[word]['deselected'][:desel_max]
        self.sel = torch.FloatTensor(
                [E.v(l) for l in self.sel]).requires_grad_(True)
        
        if len(self.desel) == 0:
            self.desel = False 
        else:
            self.desel = torch.FloatTensor(
                    [E.v(l) for l in self.desel]).requires_grad_(True)
        self.X = nn.Parameter(X)
        self.E = E
        self.g = g
        self.word = word
        self.ws = ws

    def forward(self):
        return  ran_objective(self.X, self.sel, self.desel, 
                    self.g, self.ws)


class RANDebias():
    """Class to perform Repulsion-Attraction-Neutralization based
    debiasing.
    """
    def __init__(self, E, g=None):
        """
        Args:
            E (WE class object): Word embeddings object.
            g (np.array): Gender Direction, if None, it is computed again.
        
        """
        self.E = E
        if g is None:
            g = get_g(E)
        self.g = g    

    def minimize(self, word, X, lr, max_epochs, *args, **kwargs):
        """minimize RANObjective using gradient optimization
        Args:
            word (str): word to debias
            X (np.array): the initialized new debiased vector
            lr (float): learning rate for gradient descent
            max_epochs (int): number of epochs
        
        """
        m = RANOpt(self.E, word, X, *args, **kwargs)
        optimizer = torch.optim.Adam(m.parameters(), lr=lr)
        for epoch in range(max_epochs):
            optimizer.zero_grad()
            out = m.forward()
            out.backward()
            optimizer.step()
        return m.X

    def get_new_word_embs(self, word, *args, **kwargs):
        X = init_vector(word, self.E).requires_grad_(True) 
        debiased_X = self.minimize(word, X, *args, **kwargs)
        return debiased_X/torch.norm(debiased_X)


    def run(self, words):
        """Run RANDebias for word list `words`
        
        Args:
            words (list): Words list to debias.
                
        """
        g = torch.Tensor(self.g)
        learning_rate = 0.01
        lambda_weights = [1/8, 6/8, 1/8]
        n_epochs = 300
        N_info = _get_N_info(self.E, words)

        new_embs = {}
        # for word in tqdm(words):
        for word in words:
            try:
                new_embs[word] = self.get_new_word_embs(word,
                                    learning_rate, n_epochs, N_info, g=g,
                                    ws=lambda_weights).detach().numpy()
            except:
                print(f"Failed for word: {word}")

        for w in new_embs:
            self.E.vecs[self.E.index[w]] = new_embs[w]
        return self.E
