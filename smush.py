import numpy as np
from sklearn.decompositions import PCA

def smush(matrix):
    pca = PCA(n_components=2)
    smushed = pca.fit_transform(matrix)
    return smushed

