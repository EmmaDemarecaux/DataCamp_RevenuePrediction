from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np


class Regressor(BaseEstimator):
    def __init__(self):
        self.reg = GradientBoostingRegressor(
            n_estimators=60, max_depth=50, learning_rate=0.094, 
            min_samples_split=400, min_samples_leaf=100, loss='lad')

    def fit(self, X, y):
        self.reg.fit(X, y)

    def predict(self, X):
        return self.reg.predict(X)
