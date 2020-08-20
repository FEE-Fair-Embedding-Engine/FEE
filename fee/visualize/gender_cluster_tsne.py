import numpy as np
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

class GCT():
    def __init__(self, E, random_state=0):
        self.E = E
        self.random_state = random_state 
        pass

    def cluster(self, vecs):
        labels = KMeans(n_clusters=2, 
                        random_state=self.random_state
                        ).fit_predict(vecs)
        return labels                
    
    def visualize(self, vecs, words, labels, title, figsize, dpi, colors):
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        X_embedded = TSNE(n_components=2, 
                          random_state=self.random_state
                          ).fit_transform(vecs)
        for i, (x, l, w) in enumerate(zip(X_embedded, labels, words)):
            if l:
                ax.scatter(x[0], x[1], marker = '.', c = colors[0])
            else:
                ax.scatter(x[0], x[1], marker = 'x', c = colors[1])
            ax.annotate(w, (x[0], x[1]))  
        if title is not None:
            plt.title(title)
        plt.show()
        return True    
    
    
    def run(self, word_list, title=None, dpi=300, 
            figsize=(8, 5), colors=['k', 'r']):
        vecs = [self.E.v(w) for w in word_list]    
        labels = self.cluster(vecs)
        self.visualize(vecs, word_list, labels, title, figsize, 
                        dpi, colors)