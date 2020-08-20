import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from ..utils import get_g, cosine
import matplotlib as mpl
from sklearn.manifold import TSNE

def color_fader(c1, c2, mix=0): 
    c1=np.array(mpl.colors.to_rgb(c1))
    c2=np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1-mix)*c1 + mix*c2)

def generate_palette(c1, c2, n):
    return [color_fader(c1, c2, m) for m in np.linspace(0, 1, n)]    

class NeighbourPlot():
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
        return list(bias_dict.keys())            

    def visualize(self, words, ranks, title, figsize, 
                    dpi, colors, s, annotate=False):
        assert len(words) == len(colors), "Size mismatch in colors and words"
        vecs = [self.E.v(w) for w in words]
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        X_embedded = TSNE(n_components=2, 
                          random_state=self.random_state
                          ).fit_transform(vecs)
        for i, (x, w) in enumerate(zip(X_embedded, words)):
            if not i: #First neighbour is the word itself
                ax.scatter(x[0], x[1], marker = 'x', s=s*4, c = 'black')
                ax.annotate(w, (x[0], x[1]), size=s*2)
                continue
            ax.scatter(x[0], x[1], marker = '.', c = colors[i])
            if annotate:
                ax.annotate(f"${w}_{{{ranks[w]}}}$", (x[0], x[1]), size=s)  
        if title is not None:
            plt.title(title)
        plt.show()
        return True 

    def run(self, word, title=None, n=100, dpi=300, figsize=(8, 5), 
            colors=['blue', 'red'], fontsize=7, annotate=True):
        neighbours = self.get_neighbours(word, n)  
        ranks = dict(zip(neighbours, [i for i in range(len(neighbours))]))   
        neighbours_sorted = [word] + self.bias_by_projection_sort(neighbours)  
        colors = generate_palette(colors[0], colors[1], n)
        self.visualize(neighbours_sorted, ranks, title, figsize, dpi, colors, fontsize, annotate)