import numpy as np

class RidgeScratch:
    def __init__(self, lam):
        self.lam = lam

    def fit(self, X, y):
        X = np.c_[np.ones((X.shape[0], 1)), X]
        d = X.shape[1]
        I = np.eye(d)
        I[0, 0] = 0
        self.w = np.linalg.inv(X.T @ X + self.lam * I) @ X.T @ y

    def predict(self, X):
        X = np.c_[np.ones((X.shape[0], 1)), X]
        return X @ self.w