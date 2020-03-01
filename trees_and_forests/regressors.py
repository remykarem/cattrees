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


class AdaBoostRegressor:
    def __init__(self, n_trees, max_depth=1,
                 features_to_select="all",
                 splits_to_select="all"):
        self.alpha = 0.1
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.splits_to_select = splits_to_select
        self.features_to_select = features_to_select
        self.boosted_trees = []
        self.trees_weights = []

    def fit(self, X, y):

        # 0. Initialise weights to be 1/n_rows
        n_rows = X.shape[0]
        data_weights = np.ones(shape=(n_rows,1))/n_rows

        for _ in range(self.n_trees):

            # 1. Multiply datapoints by their weights
            X = X * data_weights

            # 1. Plant a tree
            tree = plant_tree(
                Node=RegressionTreeNode,
                X=X, y=y_new, categorical=[2],
                max_depth=self.max_depth,
                features_to_select=self.features_to_select,
                splits_to_select=self.splits_to_select)
            self.boosted_trees.append(tree)

            # 2. Get predictions
            y_pred = np.array([tree(x) for x in X])

            # 3. Compare predictions and assign a weight to this tree.
            # The more incorrect predictions, the lower the weight.
            mask_errors = y_pred != y
            n_errors = sum(mask_errors) / len(y)
            tree_weight = self.alpha * np.log((1-e)/e)
            self.trees_weights.append(tree_weight)

            # 4. Update the weights of the datapoints.
            # Datapoints correctly predicted do not need to be updated.
            data_coefficients = tree_weight * mask_errors[:,None]
            data_weights = data_weights * np.exp(data_coefficients)

        self.trees_weights = np.array(self.trees_weights)

    def predict(self, X):
        preds = np.array([self.boosted_trees(x) for x in X])
        return (preds * self.trees_weights).sum()
