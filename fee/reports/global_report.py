import numpy as np
import pandas as pd
from ..utils import get_g
import seaborn as sns
import matplotlib.pyplot as plt

class GlobalReport():
    def __init__(self, E, g=None):
        if g is None:
            g = get_g(E)
        assert len(g) == E.dim   
        self.g = g
        self.E = E
    
    def plot(self, values):
        sns.distplot(values)
        plt.title("Distribution plot of bias by projection for all words.")
        plt.show()    

    def get_values_and_words(self):     
        dbs = np.abs(self.E.vecs.dot(self.g))
        sorted_values, indices = np.sort(dbs)[::-1], np.argsort(dbs)[::-1]
        sorted_words = [self.E.words[i] for i in indices]      
        return  sorted_words, sorted_values

    def print_df(self, sorted_values, sorted_words, n): 
        most_gendered_df = pd.DataFrame()
        least_gendered_df = pd.DataFrame()
        most_gendered_df['words'] = sorted_words[:n]
        most_gendered_df['bias by projection'] = sorted_values[:n]
        least_gendered_df['words'] = sorted_words[-n:]
        least_gendered_df['bias by projection'] = sorted_values[-n:]  
        print(most_gendered_df, "\n\n", least_gendered_df)    
        

    def generate(self, n=10):
        sorted_words, sorted_values = self.get_values_and_words()
        self.print_df(sorted_values, sorted_words, n)
        self.plot(sorted_values)