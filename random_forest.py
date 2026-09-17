import numpy as np
from collections import Counter
from decision_tree import DecisionTree 

def bootstrap_sample(X, y):
    n_samples = X.shape[0]
    indices = np.random.choice(n_samples, n_samples, replace=True)
    return X[indices], y[indices]

def most_common_label(y):
    return Counter(y).most_common(1)[0][0]

class RandomForest:
    def __init__(self, n_trees=10, min_samples_split=2, max_depth=100, n_feats=None):
        self.n_trees = n_trees
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTree(
                min_samples_split=self.min_samples_split,
                max_depth=self.max_depth,
                n_feats=self.n_feats
            )
            X_sample, y_sample = bootstrap_sample(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        # Transpose to shape (n_samples, n_trees)
        tree_preds = tree_preds.T
        return np.array([most_common_label(preds) for preds in tree_preds])

# Testing it
if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split

    def accuracy(y_true, y_pred):
        return np.mean(y_true == y_pred)

    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForest(n_trees=3, max_depth=10)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print("Accuracy:", accuracy(y_test, preds))
