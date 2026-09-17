import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.001, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            linear = np.dot(X, self.weights) + self.bias
            y_pred = self._sigmoid(linear)

            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        linear = np.dot(X, self.weights) + self.bias
        probs = self._sigmoid(linear)
        return np.array([1 if p > 0.5 else 0 for p in probs])

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

# Testing it
if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split

    def accuracy(y_true, y_pred):
        return np.mean(y_true == y_pred)

    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(learning_rate=0.0001, n_iters=1000)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print("Accuracy:", accuracy(y_test, preds))
