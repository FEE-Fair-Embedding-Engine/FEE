import numpy as np
import matplotlib.pyplot as plt
from ..utils import get_g, cosine
import matplotlib as mpl
from wordcloud import WordCloud   


class NBWordCloud():
    def __init__(self, E, g=None, random_state=42):
        self.E = E
        if g is None:
            g = get_g(E)
        self.g = g    
        self.random_state = random_state
    
    def get_neighbours(self, word, n=100):
        ns_idx = np.argsort(self.E.vecs.dot(self.E.v(word)))[-n:-1][::-1]
        return [self.E.words[i] for i in ns_idx]
    
    def bias_by_projection_sort(self, words):
        bias_dict = {}
        for w in words:
            bias_dict[w] = cosine(self.E.v(w), self.g)
        bias_dict = {k: v for k, v in sorted(bias_dict.items(), 
                key=lambda item: item[1])}     
        return bias_dict       

    def bias_score_to_freq(self, d):
        for k in d:
            d[k] *= 1000
            d[k] = int(d[k])
        return d            

    def visualize(self, freq_dict, title, figsize, dpi, width, height):
        wordcloud = WordCloud(width=width, height=height).generate_from_frequencies(freq_dict)
        plt.figure(figsize=figsize, dpi=dpi)
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        if title is not None:
            plt.title(title)
        plt.show()
        return True 

    def run(self, word, title=None, n=100, dpi=300, figsize=(8, 5),
                width=800, height=500):
        neighbours = self.get_neighbours(word, n)  
        neighbours_sorted_dict = self.bias_by_projection_sort(neighbours)  
        neighbours_with_freq = self.bias_score_to_freq(neighbours_sorted_dict)
        self.visualize(neighbours_with_freq, title, figsize, dpi, width, height)