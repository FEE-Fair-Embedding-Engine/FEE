import numpy as np

class HSRDebias():
    """HSR Debiasing class (Half Sibling Regression)
    """
    def __init__(self, E):
        """HSR debiasing method class. 
        
        Args:
            E (WE class object): Word embeddings object.
        
        """
        self.E = E
        
    def subset(self, words):
        """Create subset such that the words exist in vocabulary. 
        
        Args:
            words (list): list of words to debias.
        
        """        
        sub = []
        for w in words:
            try:
                sub.append(self.E.v(w))
            except:
                continue    
        return np.array(sub).T
    
    def hsr(self, gender_vecs, nongender_vecs, nongender_list, alpha):
        """Half Sibling Regression method
        
        Args:
            gender_vecs (np.array): 2D numpy array of gendered words
            nongender_vecs (np.array): 2D numpy array of non-gendered words
            nongender_list (list): list of nongender words.
            alpha (float): alpha hyperparameter.

        """
        W = np.linalg.inv(gender_vecs.T @ gender_vecs + alpha * np.eye(gender_vecs.shape[1])) @ gender_vecs.T @ nongender_vecs

        preds = gender_vecs @ np.array(W)
        deb_vecs = nongender_vecs - preds

        for i, w in enumerate(nongender_list):
            self.E.vecs[self.E.index[w]] = deb_vecs[:, i]
        return self.E
    
    def run(self, gender_list, nongender_list=None, alpha=60):
        """Run the Half Sibling Regression method
        
        Args:
            gender_list (list): list of gendered words.
            nongender_list (list): list of nongendered words.
            alpha (float): alpha hyperparameter.
                                        
        """        
        if nongender_list is None:
            nongender_list = list(set(self.E.words) - set(gender_list))
        gender_vecs = self.subset(gender_list)
        nongender_vecs = self.subset(nongender_list)

        return self.hsr(gender_vecs, nongender_vecs, 
                        nongender_list, alpha)