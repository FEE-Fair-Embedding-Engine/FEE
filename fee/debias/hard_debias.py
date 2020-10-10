import numpy as np
from ..utils import get_g

EQ_PAIRS = [
    ["monastery", "convent"],
    ["spokesman", "spokeswoman"],
    ["Catholic_priest", "nun"],
    ["Dad", "Mom"],
    ["Men", "Women"],
    ["councilman", "councilwoman"],
    ["grandpa", "grandma"],
    ["grandsons", "granddaughters"],
    ["prostate_cancer", "ovarian_cancer"],
    ["testosterone", "estrogen"],
    ["uncle", "aunt"],
    ["wives", "husbands"],
    ["Father", "Mother"],
    ["Grandpa", "Grandma"],
    ["He", "She"],
    ["boy", "girl"],
    ["boys", "girls"],
    ["brother", "sister"],
    ["brothers", "sisters"],
    ["businessman", "businesswoman"],
    ["chairman", "chairwoman"],
    ["colt", "filly"],
    ["congressman", "congresswoman"],
    ["dad", "mom"],
    ["dads", "moms"],
    ["dudes", "gals"],
    ["ex_girlfriend", "ex_boyfriend"],
    ["father", "mother"],
    ["fatherhood", "motherhood"],
    ["fathers", "mothers"],
    ["fella", "granny"],
    ["fraternity", "sorority"],
    ["gelding", "mare"],
    ["gentleman", "lady"],
    ["gentlemen", "ladies"],
    ["grandfather", "grandmother"],
    ["grandson", "granddaughter"],
    ["he", "she"],
    ["himself", "herself"],
    ["his", "her"],
    ["king", "queen"],
    ["kings", "queens"],
    ["male", "female"],
    ["males", "females"],
    ["man", "woman"],
    ["men", "women"],
    ["nephew", "niece"],
    ["prince", "princess"],
    ["schoolboy", "schoolgirl"],
    ["son", "daughter"],
    ["sons", "daughters"],
    ["twin_brother", "twin_sister"]
 ]

def _hard_neutralize(v, g):
    """Remove the gender component from a word vector.
    
    Args:
        v (np.array): Word vector.
        g (np.array): Gender Direction.
    
    Return:
        np.array: return the neutralized embedding
    
    """
    return v - g * v.dot(g) / g.dot(g)    

class HardDebias():
    """Hard debiasing class.
    
    """
    def __init__(self, E, g=None):
        """HardDebias debiasing method class. 
        
        This debiasing word vectors in two step
        stages, first it neutralizes and then equailizes the vectors.
        
        Args:
            E (WE class object): Word embeddings object.
            g (np.array): Gender Direction, if None, it is computed again.
        
        """
        if g is None:
            g = get_g(E)
        assert len(g) == E.dim   
        self.g = g
        self.E = E
        self.equalize_pairs = EQ_PAIRS

    
    def neutralize(self, E, word_list):
        """Neutralize word vectors using the gender direction. This is the 
        first step of hard debiasing procedure.
        
        Args:
            E (WE class object): Word embeddings object.
            word_list (list): List of words to debias.
        
        """        
        for i, w in enumerate(E.words):
            if w in word_list:
                E.vecs[i] = _hard_neutralize(E.vecs[i], self.g)
        return E    
    
    def equalize(self, E):
        """Equalize word vectors using the gender direction and a set of 
        equalizing word pairs. This is the second step of hard debiasing 
        procedure.
        
        Args:
            E (WE class object): Word embeddings object.
        
        """                
        g = self.g
        candidates = {x for e1, e2 in self.equalize_pairs for x in [
                                        (e1.lower(), e2.lower()),
                                        (e1.title(), e2.title()),
                                        (e1.upper(), e2.upper())]
                    }
        for (a, b) in candidates:
            if (a in E.index and b in E.index):
                y = _hard_neutralize((E.v(a) + E.v(b)) / 2, g)
                z = np.sqrt(1 - np.linalg.norm(y)**2)
                if (E.v(a) - E.v(b)).dot(g) < 0:
                    z = -z
                E.vecs[E.index[a]] = z * g + y
                E.vecs[E.index[b]] = -z * g + y
        return E        
        
    
    def run(self, word_list):
        """Debias word vectors using the hard debiasing method. 

        Args:
            word_list (list): List of words to debias.
    
        Return:
            Debiased word vectors
        """             
        return self.equalize(self.neutralize(self.E, word_list))