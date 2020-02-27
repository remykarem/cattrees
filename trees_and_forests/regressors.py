import numpy as np
from .tree import plant_tree
from .data_structures import RegressionTreeNode
import ipdb

__all__ = ["DecisionTreeRegressor"]

X = np.array([
    [1, 101, 103, 3, 5, 107, 109, 7, 8],
    [1, 101, 103, 3, 5, 107, 109, 7, 8],
    [0,   0,   0, 0, 0,   0,   0, 1, 1]]).T
y = np.array([13, 30, 22, 31, 16, 19, 18, 27, 15])


class DecisionTreeRegressor:
    def __init__(self):
        self.tree = None

    def fit(self, X, y, categorical=[2]):
        """Build a decision tree to fit X and y"""
        self.tree = plant_tree(RegressionTreeNode, X, y)

    def predict(self, X):
        return self.tree(X)
