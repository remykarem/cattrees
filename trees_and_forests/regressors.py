import numpy as np
from .tree import plant_tree, plant_tree_average
from .data_structures import RegressionTreeNode

__all__ = ["DecisionTreeRegressor", "GradientBoostingRegressor"]


class DecisionTreeRegressor:
    def __init__(self,
                 max_depth=3,
                 features_to_select="all",
                 splits_to_select="all"):
        self.tree = None
        self.max_depth = max_depth
        self.splits_to_select = splits_to_select
        self.features_to_select = features_to_select

    def fit(self, X, y, categorical=[2]):
        """Build a decision tree to fit X and y"""
        self.tree = plant_tree(
            Node=RegressionTreeNode,
            X=X, y=y, categorical=categorical,
            max_depth=self.max_depth,
            features_to_select=self.features_to_select,
            splits_to_select=self.splits_to_select)

    def predict(self, X):
        return [self.tree(x[None, :]) for x in X]


class GradientBoostingRegressor:
    def __init__(self, n_trees, max_depth=1,
                 features_to_select="all",
                 splits_to_select="all"):
        self.alpha = 0.1
        self.n_trees = n_trees
        self.tree_boosted = None
        self.max_depth = max_depth
        self.splits_to_select = splits_to_select
        self.features_to_select = features_to_select

    def fit(self, X, y):
        tree = plant_tree_average(RegressionTreeNode, y)
        y_new = y
        for _ in range(2):
            y_now = np.array([tree(x) for x in X])
            y_new = y_now + self.alpha * (y_new - y_now)
            tree = plant_tree(
                Node=RegressionTreeNode,
                X=X, y=y_new, categorical=[2],
                max_depth=self.max_depth,
                features_to_select=self.features_to_select,
                splits_to_select=self.splits_to_select)
        self.tree_boosted = tree

    def predict(self, X):
        return np.array([self.tree_boosted(x) for x in X])
