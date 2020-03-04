import numpy as np
from .tree import plant_tree, plant_tree_average
from .data_structures import RegressionTreeNode


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


class RandomForestRegressor:
    def __init__(self,
                 n_trees=10,
                 bootstrap_samples=True,
                 max_depth=3,
                 features_to_select="sqrt",
                 splits_to_select="all"):
        self.trees = None
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.splits_to_select = splits_to_select
        self.features_to_select = features_to_select

    def fit(self, X, y, categorical=[2]):
        n_samples = X.shape[0]
        self.trees = [
            plant_tree(
                Node=RegressionTreeNode,
                X=X[get_bootstrap_sample_indices(n_samples), :],
                y=y,
                categorical=categorical,
                max_depth=self.max_depth,
                features_to_select=self.features_to_select,
                splits_to_select=self.splits_to_select)
            for i in range(self.n_trees)]

    def predict(self, X):
        preds = np.array([tree(X) for tree in self.trees])
        max_voting = np.argmax(np.bincount(preds))
        return max_voting


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

        # 0a. Start with a simple tree
        tree = plant_tree_average(RegressionTreeNode, y)

        # TODO
        # 0b. (?)
        y_new = y

        for _ in range(self.n_trees):

            # 1. Get current predictions based on this tree
            y_now = np.array([tree(x) for x in X])

            # TODO
            # 2. Based on gradient descent, update to get the new
            y_new = y_now + self.alpha * (y_new - y_now)

            # 3. Plant tree that tries to fit (X,y_new)
            tree = plant_tree(
                Node=RegressionTreeNode,
                X=X, y=y_new, categorical=[2],
                max_depth=self.max_depth,
                features_to_select=self.features_to_select,
                splits_to_select=self.splits_to_select)

        self.tree_boosted = tree

    def predict(self, X):
        return np.array([self.tree_boosted(x) for x in X])

