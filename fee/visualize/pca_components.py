import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

class PCAComponents():
    """`PCAComponents` Class"""
    def __init__(self, E):
        """Plot the PCA principle component bar graph for some direction 
        of `E` computed using a list of pairs of words.
        
        Args:
            E (WE class object): Word embeddings object
        
        """               
        self.E = E

    def PlotPCA(self, pairs, title, num_components, dpi, figsize):
        """Main `PCAComponents` visualization driver function
        
        Args:
            pairs (list): A list of pair (tuple/list) of words. The 
                        direction is computed by PCA of set of 
                        differences of these words. 
            title (str): title of the plot
            num_components (int): number of principal components
            dpi (int): dpi of the figures  
            figsize (tuple): size of figures in (HxW)  
        
        """           
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

    def run(self, pairs, title=None, num_components=10, dpi=300, 
            figsize=(8, 5)):
        """Run the `PCAComponents` visualization
        
        Args:
            pairs (list): A list of pair (tuple/list) of words. The 
                        direction is computed by PCA of set of 
                        differences of these words. 
            title (str): title of the plot
            num_components (int): number of principal components
            figsize (tuple): size of figures in (HxW)  
            dpi (int): dpi of the figures  
        
        """              
        assert len(pairs) >= num_components, f"# pairs ({len(pairs)}) should be greater than the number of components ({num_components})."
        self.PlotPCA(pairs, title, num_components, dpi, figsize)        