import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

class PCAComponents():
    def __init__(self, E):
        self.E = E

    def PlotPCA(self, pairs, title, num_components, dpi, figsize):
        plt.figure(figsize=figsize, dpi=dpi)
        matrix = []
        for a, b in pairs:
            center = (self.E.v(a) + self.E.v(b))/2
            matrix.append(self.E.v(a) - center)
            matrix.append(self.E.v(b) - center)
        matrix = np.array(matrix)
        pca = PCA(n_components = num_components)
        pca.fit(matrix)
        plt.bar(range(num_components), pca.explained_variance_ratio_)
        if title is not None:
            plt.title(title)
        plt.show()

    def run(self, pairs, title=None, num_components=10, dpi=300, figsize=(8, 5)):
        self.PlotPCA(pairs, title, num_components, dpi, figsize)        