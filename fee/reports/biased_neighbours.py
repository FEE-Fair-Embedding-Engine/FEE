import numpy as np
from ..utils import get_g, cosine
import pandas as pd

class NeighboursAnalysis():
    def __init__(self, E, g=None, random_state=42):
        """Analyze Neighbours of a word in the embedding through 
        cosine similarities and bias by projection.
        Args:
            E (WE class object): Word embeddings object
        Kwargs:
            g (np.array): gender direction
            random_state (int): random seed for reproduction
        
        """
        self.E = E
        if g is None:
            g = get_g(E)
        self.g = g    
        self.random_state = random_state
    
    def get_neighbours(self, word, n=100):
        """Compute list of `n` neighbours for `word`
        Args:
            word (str): Word to compute neighbours for
        Kwargs:
            n (int): number of neighbours to compute
        
        """        
        ns_idx = np.argsort(self.E.vecs.dot(self.E.v(word)))[-n:-1][::-1]
        return [self.E.words[i] for i in ns_idx]
    
    def print_neighbours(self, words, n):
        """Pretty print `n` neighbours 
        Args:
            words (list): List of neighbours
        Kwargs:
            n (int): number of neighbours to compute
        
        """          
        bias_dict = {}
        for w in words:
            bias_dict[w] = cosine(self.E.v(w), self.g)
        bias_dict = {k: v for k, v in sorted(bias_dict.items(), 
                key=lambda item: item[1])}     
        report_df = pd.DataFrame()
        report_df["Neighbour"] = list(bias_dict.keys())[::-1][:n]                
        report_df["Bias by projection"] = list(bias_dict.values())[::-1][:n] 
        return report_df            
           
    def generate(self, word, n=100, ret_report=True):
        """Generate the report for neighbours of word
        Args:
            word (str): Word to generate report for
        Kwargs:
            n (int): number of neighbours to compute
            ret_report (bool): return or print the report dataframe
        
        """            
        neighbours = self.get_neighbours(word, n)  
        report_df = self.print_neighbours(neighbours, n)
        if ret_report:
            return report_df
        else:
            print(report_df)
              
