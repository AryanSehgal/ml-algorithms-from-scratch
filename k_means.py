import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

class KMeans:
    def __init__(self, K=3, max_iters=100, plot_steps=False):
        self.K = K
        self.max_iters = max_iters
        self.plot_steps = plot_steps
        self.clusters = [[] for _ in range(self.K)]
        self.centroids = []

    def predict(self, X):
        self.X = X
        self.n_samples, self.n_features = X.shape

        # Initialize centroids
        random_idxs = np.random.choice(self.n_samples, self.K, replace=False)
        self.centroids = [X[idx] for idx in random_idxs]

        for _ in range(self.max_iters):
            # Assign clusters
            self.clusters = self._create_clusters(self.centroids)

            if self.plot_steps:
                self.plot()

            # Update centroids
            centroids_old = self.centroids
            self.centroids = self._calculate_centroids(self.clusters)

            # Check convergence
            if self._is_converged(centroids_old, self.centroids):
                break

        return self._get_cluster_labels(self.clusters)

    def _create_clusters(self, centroids):
        clusters = [[] for _ in range(self.K)]
        for idx, sample in enumerate(self.X):
            closest_idx = self._closest_centroid(sample, centroids)
            clusters[closest_idx].append(idx)
        return clusters

    def _closest_centroid(self, sample, centroids):
        distances = [euclidean_distance(sample, point) for point in centroids]
        return np.argmin(distances)

    def _calculate_centroids(self, clusters):
        centroids = np.zeros((self.K, self.n_features))
        for idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis=0)
            centroids[idx] = cluster_mean
        return centroids

    def _is_converged(self, old_centroids, new_centroids):
        distances = [
            euclidean_distance(old_centroids[i], new_centroids[i])
            for i in range(self.K)
        ]
        return np.sum(distances) == 0

    def _get_cluster_labels(self, clusters):
        labels = np.empty(self.n_samples)
        for cluster_idx, sample_idxs in enumerate(clusters):
            labels[sample_idxs] = cluster_idx
        return labels

    def plot(self):
        fig, ax = plt.subplots(figsize=(10, 6))
        for i, cluster in enumerate(self.clusters):
            points = self.X[cluster].T
            ax.scatter(*points)
        for point in self.centroids:
            ax.scatter(*point, marker='x', color='black', linewidth=2)
        plt.show()

# Testing
if __name__ == "__main__":
    from sklearn.datasets import make_blobs

    X, y = make_blobs(n_samples=500, centers=3, n_features=2, random_state=40)
    kmeans = KMeans(K=3, max_iters=150, plot_steps=True)
    y_pred = kmeans.predict(X)
    kmeans.plot()
