import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ..utils import get_g
from ..metrics import DirectBias, ProxBias
from ..visualize import NeighbourPlot, NBWordCloud
from .biased_neighbours import NeighboursAnalysis
 

class WordReport():
    def __init__(self, E, g=None):
        if g is None:
            g = get_g(E)
        assert len(g) == E.dim   
        self.g = g
        self.E = E
    
    def generate(self, word, n=50, figsize=None, dpi=100):
        direct_bias = DirectBias(self.E).compute(word)
        prox_bias = ProxBias(self.E).compute(word)
        neighbours_bias = NeighboursAnalysis(self.E).generate(word, n=n)

        print("==============================")
        print(f"Direct bias (Bias by projection on the PCA based gender direction): {direct_bias}")
        print("==============================")
        print(f"Proximity bias (Ratio of biased neighbours by Indirect Bias): {prox_bias}")
        print("==============================")
        print(f"Neighbour Analysis: \n{neighbours_bias}")
        print("==============================")
        
        NeighbourPlot(self.E).run(word, annotate=True, 
                        figsize=figsize, dpi=dpi, n=n,
                        title=f"tSNE plot for neighbours of {word} (color coded by bias by projection)")
        NBWordCloud(self.E).run(word, figsize=figsize, dpi=dpi,
                        title=f"Wordcloud for neighbours of {word} (size coded by bias by projection)")
