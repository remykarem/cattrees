import numpy as np
from ._tree import plant_tree

X = np.array([
    [1, 101, 103, 3, 5, 107, 109, 7, 8],
    [1, 101, 103, 3, 5, 107, 109, 7, 8],
    [0,   0,   0, 0, 0,   0,   0, 1, 1]]).T
y = np.array([0, 1, 1, 0, 0, 1, 1, 2, 2])


class DecisionTreeClassifier:
    def __init__(self):
        self.tree = None

    def fit(self, X, y, categorical=[2]):
        """Build a decision tree to fit X and y"""
        self.tree = plant_tree(X, y)

    def predict(self, X):
        return self.tree(X)


class RandomForestClassifier:
    def __init__(self, n_trees=10):
        self.n_trees = n_trees
        self.trees = None

    def fit(self, X, y, categorical=[2]):
        """Build a decision tree to fit X and y"""
        self.trees = [plant_tree(X, y, features_to_select="sqrt")
                      for i in range(self.n_trees)]

    def predict(self, X):
        preds = np.array([tree(X) for tree in self.trees])
        max_voting = np.argmax(np.bincount(preds))
        return max_voting
