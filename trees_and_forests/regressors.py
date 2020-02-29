from .tree import plant_tree
from .data_structures import RegressionTreeNode

__all__ = ["DecisionTreeRegressor"]


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
        return self.tree(X)
