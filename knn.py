import numpy as np
from collections import Counter

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        return np.array([self._predict(x) for x in X])

    def _predict(self, x):
        # Compute distances to all training points
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        # Get indices of k nearest points
        k_indices = np.argsort(distances)[:self.k]
        # Fetch the labels of those points
        k_labels = [self.y_train[i] for i in k_indices]
        # Return the most common label
        return Counter(k_labels).most_common(1)[0][0]

# Testing
if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from matplotlib.colors import ListedColormap

    def accuracy(y_true, y_pred):
        return np.mean(y_true == y_pred)

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = KNN(k=3)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print("KNN Classification Accuracy:", accuracy(y_test, preds))
