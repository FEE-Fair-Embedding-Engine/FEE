import numpy as np
from sklearn.decomposition import PCA

def doPCA(pairs, embedding, num_components = 10, plot=False):
    matrix = []
    for a, b in pairs:
        center = (embedding.v(a) + embedding.v(b))/2
        matrix.append(embedding.v(a) - center)
        matrix.append(embedding.v(b) - center)
    matrix = np.array(matrix)
    pca = PCA(n_components = num_components)
    pca.fit(matrix)
    if plot:
        bar(range(num_components), pca.explained_variance_ratio_)
    return pca

def get_g(E, definitional=None):
    if definitional is None:
        definitional = [['woman', 'man'],
                        ['girl', 'boy'],
                        ['she', 'he'],
                        ['mother', 'father'],
                        ['daughter', 'son'],
                        ['gal', 'guy'],
                        ['female', 'male'],
                        ['her', 'his'],
                        ['herself', 'himself'],
                        ['mary', 'john']] #Source: tolga
    g = doPCA(definitional, E).components_[0]
    return g 

def cosine(v1, v2):
    norm_p = np.linalg.norm(v1) * np.linalg.norm(v2)
    return np.abs(v1.dot(v2)) / norm_p   

def get_pair_idb(w, v, g, E):
    if isinstance(w, str): w = E.v(w)
    if isinstance(v, str): v = E.v(v)
    w_orth = w - (np.dot(w, g)) * g
    v_orth = v - (np.dot(v, g)) * g
    dots = np.dot(w, v)
    orth_dots = np.dot(w_orth, v_orth)
    idb = (dots - orth_dots / (np.linalg.norm(w_orth) * np.linalg.norm(v_orth))) / (dots )
    return idb     
    
def get_nbs(E, word, k=100):
    return np.argsort(E.vecs.dot(E.v(word)))[-k:][::-1]


