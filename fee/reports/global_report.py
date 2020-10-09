import numpy as np
import pandas as pd
from ..utils import get_g
import seaborn as sns
import matplotlib.pyplot as plt

class GlobalReport():
    def __init__(self, E, g=None):
        """Generate a global bias report for a word embedding. This
        report computes the least and most biased words in an embedding
        and plot them. Bias by projection (direct bias) is used as the 
        metric to compute this report. The report also plots the overall
        distribution of bias in the embedding `E`.
        Args:
            E (WE class object): Word embeddings object
        Kwargs:
            g (np.array): gender direction
        """        
        if g is None:
            g = get_g(E)
        assert len(g) == E.dim   
        self.g = g
        self.E = E
    
    def plot(self, values):
        """Plot the biased words.
        Args:
            values (list): list of bias by projection
        """
        sns.distplot(values)
        plt.title("Distribution plot of bias by projection for all words.")
        plt.show()    

    def get_values_and_words(self):     
        """Get the list of words in `E` sorted by bias by projection. 
        """
        dbs = np.abs(self.E.vecs.dot(self.g))
        sorted_values, indices = np.sort(dbs)[::-1], np.argsort(dbs)[::-1]
        sorted_words = [self.E.words[i] for i in indices]      
        return  sorted_words, sorted_values

    def print_df(self, sorted_values, sorted_words, n): 
        """Pretty print the dataframe containing most and least biased
        words in `E`.
        Args:
            sorted_words (list): list of bias by projection for 
                                 `sorted_words`
            sorted_words (list): list of words
            n (int): no. of least/most biased words to print
        """        
        most_gendered_df = pd.DataFrame()
        least_gendered_df = pd.DataFrame()
        most_gendered_df['words'] = sorted_words[:n]
        most_gendered_df['bias by projection'] = sorted_values[:n]
        least_gendered_df['words'] = sorted_words[-n:]
        least_gendered_df['bias by projection'] = sorted_values[-n:]  
        print(most_gendered_df, "\n\n", least_gendered_df)    
        

    def generate(self, n=10):
        """Generate the global report for embedding `E`
        Args:
            n (int): No. of most/least biased words to print.
        """
        sorted_words, sorted_values = self.get_values_and_words()
        self.print_df(sorted_values, sorted_words, n)
        self.plot(sorted_values)