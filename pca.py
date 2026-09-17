import numpy as np

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        # Center the data
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # Compute covariance matrix
        cov_matrix = np.cov(X_centered.T)

        # Get eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

        # Sort eigenvectors by descending eigenvalues
        sorted_idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, sorted_idx]

        # Keep only top n components
        self.components = eigenvectors[:, :self.n_components].T

    def transform(self, X):
        X_centered = X - self.mean
        return np.dot(X_centered, self.components.T)

# Testing it
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from sklearn.datasets import load_iris

    X, y = load_iris(return_X_y=True)

    pca = PCA(n_components=2)
    pca.fit(X)
    X_reduced = pca.transform(X)

    print("Original shape:", X.shape)
    print("Reduced shape:", X_reduced.shape)

    plt.scatter(
        X_reduced[:, 0],
        X_reduced[:, 1],
        c=y,
        cmap=plt.cm.get_cmap("viridis", 3),
        alpha=0.8,
        edgecolors="none"
    )
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.colorbar()
    plt.show()
