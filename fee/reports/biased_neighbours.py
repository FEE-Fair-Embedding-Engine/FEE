import numpy as np
from ..utils import get_g, cosine
import pandas as pd

class NeighboursAnalysis():
    def __init__(self, E, g=None, random_state=42):
        self.E = E
        if g is None:
            g = get_g(E)
        self.g = g    
        self.random_state = random_state
    
    def get_neighbours(self, word, n=100):
        ns_idx = np.argsort(self.E.vecs.dot(self.E.v(word)))[-n:-1][::-1]
        return [self.E.words[i] for i in ns_idx]
    
    def print_neighbours(self, words, n):
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
        neighbours = self.get_neighbours(word, n)  
        report_df = self.print_neighbours(neighbours, n)
        if ret_report:
            return report_df
        else:
            print(report_df)
              
