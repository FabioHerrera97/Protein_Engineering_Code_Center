from sklearn.decomposition import PCA

class PCA_reduction:
    
    def __init__(self, n_components):
        self.pca = PCA(n_components=n_components)
    
    def fit_transform(self, X):
        return self.pca.fit_transform(X)
    
    def transform(self, X):
        return self.pca.transform(X)