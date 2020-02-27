import numpy as np
from .tree import plant_tree
from .data_structures import ClassificationTreeNode
import ipdb

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
        self.tree = plant_tree(ClassificationTreeNode, X, y)

    def predict(self, X):
        return self.tree(X)


class RandomForestClassifier:
    def __init__(self, n_trees=10, bootstrap_samples=True):
        self.n_trees = n_trees
        self.trees = None

    def fit(self, X, y, categorical=[2]):
        n_rows = X.shape[0]
        self.trees = [
            plant_tree(
                Node=ClassificationTreeNode,
                X=X[get_bootstrap_sample_indices(n_rows), :],
                y=y,
                features_to_select="sqrt")
            for i in range(self.n_trees)]

    def predict(self, X):
        preds = np.array([tree(X) for tree in self.trees])
        max_voting = np.argmax(np.bincount(preds))
        return max_voting


def get_bootstrap_sample_indices(n_rows):
    PROPORTION = 2/3
    final_size = min(1, int(PROPORTION*n_rows))
    return np.random.choice(n_rows, size=final_size, replace=True)
