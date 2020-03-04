import numpy as np
from .tree import plant_tree
from .data_structures import ClassificationTreeNode

X = np.array([
    [1, 101, 103, 3, 5, 107, 109, 7, 8],
    [1, 101, 103, 3, 5, 107, 109, 7, 8],
    [0,   0,   0, 0, 0,   0,   0, 1, 1]]).T
y = np.array([0, 1, 1, 0, 0, 1, 1, 2, 2])


class DecisionTreeClassifier:
    def __init__(self,
                 max_depth=3,
                 features_to_select="all",
                 splits_to_select="all"):
        self.tree = None
        self.max_depth = max_depth
        self.splits_to_select = splits_to_select
        self.features_to_select = features_to_select

    def fit(self, X, y, categorical=[2]):
        self.tree = plant_tree(
            Node=ClassificationTreeNode,
            X=X,
            y=y,
            categorical=categorical,
            max_depth=self.max_depth,
            features_to_select=self.features_to_select,
            splits_to_select=self.splits_to_select)

    def predict(self, X):
        return self.tree(X)


class RandomForestClassifier:
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
                Node=ClassificationTreeNode,
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


class BaggingClassifier:
    def __init__(self):
        pass

    def fit(self):
        pass

    def predict(self):
        pass


class ExtRaTreesClassifier:
    def __init__(self):
        pass

    def fit(self):
        pass

    def predict(self):
        pass


class GradientBoostingClassifier:
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
        tree = plant_tree_average(ClassificationTreeNode, y)
        y_new = y
        for _ in range(2):
            y_now = np.array([tree(x) for x in X])
            y_new = y_now + self.alpha * (y_new - y_now)
            tree = plant_tree(
                Node=ClassificationTreeNode,
                X=X, y=y_new, categorical=[2],
                max_depth=self.max_depth,
                features_to_select=self.features_to_select,
                splits_to_select=self.splits_to_select)
        self.tree_boosted = tree

    def predict(self, X):
        return np.array([self.tree_boosted(x) for x in X])


def get_bootstrap_sample_indices(n_rows, proportion=2/3):
    final_size = min(1, int(proportion*n_rows))
    return np.random.choice(n_rows, size=final_size, replace=True)


class AdaBoostRegressor:
    def __init__(self, n_trees, max_depth=1,
                 features_to_select="all",
                 splits_to_select="all"):
        self.alpha = 0.1
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.splits_to_select = splits_to_select
        self.features_to_select = features_to_select
        self.trees = None
        self.says = None

    def fit(self, X, y):

        # 0. Initialise weights to be 1/n_rows, prepare a list of say
        n_rows = X.shape[0]
        row_indices = np.arange(n_rows)
        weights = np.ones(shape=(n_rows, 1))/n_rows
        trees = []
        says = []  # amount of say for each tree

        for _ in range(self.n_trees):

            # 1. Retrieve bootstrap samples by sampling from a categorical
            # distribution with `weights` as the probabilities.
            # Once done, reset the weights
            rows_to_select = np.random.choice(row_indices, size=n_rows, p=weights.T)
            _X = X[rows_to_select]
            weights = np.ones(shape=(n_rows, 1))/n_rows

            # 2. Plant a tree
            tree = plant_tree(
                Node=RegressionTreeNode,
                X=_X, y=y_new, categorical=[2],
                max_depth=2,
                features_to_select=self.features_to_select,
                splits_to_select=self.splits_to_select)

            # 3. Calculate tree's final say during aggregation
            #    a) Calculate predictions
            #    b) Calculate error rate
            #    c) Calculate current tree's say
            y_pred = np.array([tree(x) for x in _X])
            mask_for_errors = y_pred != y
            error_rate = sum(mask_for_errors) / len(y)
            say = 0.5 * np.log((1-error_rate)/error_rate)

            # 4. Update the weights of the datapoints.
            # Datapoints correctly predicted do not need to be updated.
            exponent = say * mask_for_errors[:, None]
            weights = weights * np.exp(exponent)

            # 5. Normalise weights
            weights = weights / weights.sum()

            trees.append(tree)
            says.append(say)

        self.trees = trees
        self.says = np.array(says)

    def predict(self, X):
        preds = np.array([self.boosted_trees(x) for x in X])
        weighted_pred = (preds * says).sum()
        return weighted_preds
