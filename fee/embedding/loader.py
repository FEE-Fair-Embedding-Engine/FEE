import zipfile
import os
# from tqdl import download
from tqdm import tqdm
import re
import gc
import numpy as np
import codecs
import gensim.downloader as api
from gensim.test.utils import get_tmpfile

         
class WE():
    """The Word embedding class.

    The main class that facilitates the word embedding structure. 

    Attributes
    ----------
    dim (int): Dimension of embedding
    vecs (np.array): 

    """
    def __init__(self):
        """
        Initialize WE object.
        """
        # self.downloader = Downloader()
        self.desc = "Word embedding loader for "


    def fname_to_format(self, fname):
        """Get embedding format from file name.

        Format can usually be extracted from the filename extension. We
        currently support the loading of embeddings in binary (.bin), 
        text (.txt) and numpy format (.npy). 
        
        Args:
            fname (str): file name
        
        Return:
            format (str): format (txt, bin or npy)
        """
        if fname is None:
            raise "fname can't be None"
            return None

        if fname.endswith('.txt'):
            format = 'txt'
        elif fname.endswith('.bin'):
            format = 'bin'
        else:
            format = 'npy'
        return format            

    def get_gensim_word_vecs(self, model):
        """ Loading word and vecs using gensim scripts.
        Args:
            model (gensim object): Model for accessing all the words in 
                                   vocab, and their vectors.  
        """
        words = sorted([w for w in model.vocab], key=lambda w: model.vocab[w].index)
        vecs = np.array([model[w] for w in words])
        return words, vecs
    
    def _load(self, fname, format, dim = 300):
        """Internal load function.

        There shall be no exceptions in this function. Verify everything
        beforehand. Loads word embedding at location `fname` on disk.

        Args:
            fname (str): Path to the embedding file on disk.
            format (str): Format of word embedding. Following are the
                          supported formats:
                            - binary
                            - text
                            - numpy array
            dim (int): The dimension of embedding vectors.
        
        Return:
            words (list): List of vocabulary words.
            vecs (np.array): Word vectors of size (self.n, self.dim)     

        """
        vecs = []
        words = []
        
        if format is None:
            format = self.fname_to_format(fname)  

        if format == 'bin':
            import gensim.models
            model = gensim.models.KeyedVectors.load_word2vec_format(fname, binary=True)
            words, vecs = self.get_gensim_word_vecs(model)
            
        elif format == 'txt':
            with open(fname, "r") as f:
                lines = f.readlines()
                for line in lines:
                    tokens = line.split()
                    v = np.array([float(x) for x in tokens[-dim:]])
                    w = "_".join([str(x) for x in tokens[:-dim]])
                    if len(v) != dim:
                        print(f"Weird line: {tokens} | {len(v)}")
                        continue
                    words.append(w)
                    vecs.append(v)
        else:
            with codecs.open(fname + '.vocab', 'r', 
                            'utf-8') as f_embed:
                words = [line.strip() for line in f_embed]
            vecs = np.load(fname + '.wv.npy')
        
        self.n, self.dim = vecs.shape
        
        self.desc = f"File: {fname}\tFormat: {format}\t" \
                    f"#Words: {self.n}\tDimension: {self.dim}"
        return words, vecs    
    
    def load(self, fname=None, format=None, ename=None, 
            normalize=False, dim = 300):
        """Load word embedding from filename or embedding name.

        Loads word embeddings from either filename `fname` or the 
        embedding name `ename`. Following formats are supported:
        - bin: Binary format, load through gensim.
        - txt: Text w2v or GloVe format.
        - npy: Numpy format. `fname.wv.npy` contans the numpy vector
               while `fname.vocab` contains the vocabulary list.
        All Gensim pre-trained embeddings are integrated for easy access
        via `ename`. `ename` are same as the gensim conventions. 

        Example:
            ```
            we = WE()
            E = we.load('glove6B.txt', dim = 300)
            ```
            ```
            we = WE()
            E = we.load(ename = 'glove-wiki-gigaword-50')
            ```

        Args:
            fname (str): Path to the embedding file on disk.
            format (str): Format of word embedding. Following are the
                          supported formats:
                            - binary
                            - text
                            - numpy array
            ename (str): Name of embedding. This will download embedding
                         using the `Downloader` class. In case both 
                         ename and fname are provided, ename is given
                         priority.
            normalize (bool): Normalize word vectors or not.
            dim (int): The dimension of embedding vectors. 
                       Default dimension is 300

        Return:
            self (WE object): Return self, the word embedding object.                         
        """
        if ename is not None:
            model = api.load(ename)
            words, vecs = self.get_gensim_word_vecs(model)

        else:
            words, vecs = self._load(fname, format, dim)

        self.words = words
        self.vecs = vecs
        self.reindex()
        self.normalized = normalize
        if normalize:
            self.normalize()
        return self    

    def reindex(self):
        """Reindex word vectors.
        """
        self.index = {w: i for i, w in enumerate(self.words)}
        self.n, self.dim = self.vecs.shape
        assert self.n == len(self.words) == len(self.index)

    def v(self, word):
        """Access vector for a word

        Returns the `self.dim` dimensional vector for the word `word`.

        Example:
            E = WE().load('glove')
            test = E.v('test')

        Args:
            word (str): Word to access vector of.

        Return:
            vec (np.array): `self.dim` dimension vector for `word`.    
        """
        vec = self.vecs[self.index[word]]
        return vec

    def normalize(self):
        """Normalize word embeddings.

        Normaliation is done as follows:
            \vec{v}_{norm} := \vec{v}/|\vec{v}|
            where |\vec{v}| is the L2 norm of \vec{v}
        """
        self.vecs /= np.linalg.norm(self.vecs, axis=1)[:, np.newaxis]
        self.reindex()        
        self.desc += "\tNormalized: True"
        self.normalized = True

    def __repr__(self):
        """Class `__repr__` object for pretty informational print.
        """
        return self.desc            



# if __name__ == "__main__":
#     E = WE().load(ename = "glove-wiki-gigaword-100", normalize=True)
#     print(E.v('dog'))
#     print(E)