import warnings
import numpy as np
from .utils import *
from .exceptions import NotEvaluatedError, NotSupposedToHappenError


class TreeNode:
    """Decision tree node

    We would not want to define another node in the constructor
    so the API requires you to set the left and right
    node attributes manually.

    Params
    ------
    data: tuple of (X,y) where X and y are `ndarray`s
        Training data
    idx: int
        Node id
    depth: int
        Depth of node in the tree it is in
    """

    def __init__(self,
                 data=None, categorical_features=None,
                 idx=None,
                 depth=None):
        self.data = gen(*data)
        self.idx = idx
        self.depth = depth
        self.categorical_features = categorical_features
        self.qn = None
        self.col = None
        self.pred = None
        self.left = None
        self.right = None
        self.criteria = None
        self.evaluated = False

    def split(self, max_depth, features_to_select, splits_to_select):
        """
        Performs a split on the self.data.
        Returns none if no split should be done.
        This method makes self become 'evaluated'.

        Criterion = gini

        Returns
        -------
        left_data: ndarray
            Feature
        right_data: ndarray
            Feature
        """
        # 1. Get the features and the targets
        features, y = next(self.data)
        if features.ndim == 1:
            features = features[:, None]

        # 2. Check conditions to not split.
        # Terminal nodes must provide prediction value.
        if len(np.unique(y)) == 1:
            self.pred = np.unique(y)[0]
            self.evaluated = True
            return None, None
        elif self.depth == max_depth:
            self.pred = np.argmax(np.bincount(
                y)) if self.criteria == "gini" else np.mean(y)
            self.evaluated = True
            return None, None
        elif self.depth > max_depth:
            raise NotSupposedToHappenError

        # 3. Prepare features to loop through.
        feature_indices_to_select = select_features(
            n_features=features.shape[1], method=features_to_select)

        # 4. For categorical features, we want to perform a one-time
        # calculation of the no. of classes. This will be used later
        # when calculating the gini coefficient. Also, set it to empty
        # list if there're no categorical features.
        n_classes = np.max(y)+1 if self.criteria == "gini" else None
        if self.categorical_features is None:
            self.categorical_features = []

        # 5. Prepare arrays to store criteria and thresholds
        # from looping through features and thresholds.
        # xxx means 'category' (categorical) or 'split' (numerical).
        best_xxx_from_every_feature = [0]*features.shape[1]
        best_gain_from_every_feature = [0]*features.shape[1]

        # 6. Before looping through to get criteria, find the dataset's
        # initial criterion.
        criterion_initial = calculate_criterion_initial(self.criteria, y)
        print(f"Criterion initial: {criterion_initial}")

        # 7. Loop through every feature and threshold
        for col_num, feature in enumerate(features.T):

            if col_num not in feature_indices_to_select:
                continue

            if len(np.unique(feature)) == 1:
                # FIXME what happens if there is only one feature
                # and that feature has only 1 unique number?
                warnings.warn(
                    f"Encountered only one unique feature in col {col_num}")
                self.pred = np.argmax(np.bincount(y))
                self.evaluated = True
                return None, None

            criterion_gains_for_one_feature = []

            # 7a. Working with categorical features
            if col_num in self.categorical_features:

                print()
                print(f"X[:,{col_num}]: {feature}")
                print(f"     y: {y}")
                print()

                # i) Loop through every threshold. For categorical features,
                # thresholds are the categories themselves.
                for category in np.unique(feature):

                    left, right = y[feature ==
                                    category], y[feature != category]

                    criterion = calculate_criterion(
                        self.criteria, left, right, n_classes)
                    weights = np.array([
                        len(left)/len(feature),
                        1-len(left)/len(feature)])
                    weighted_criterion = np.sum(weights * criterion)
                    criterion_gain = criterion_initial - weighted_criterion

                    criterion_gains_for_one_feature.append(criterion_gain)
                    print(left, right)
                    print(f"Criterion gain: {criterion_gain}")

                # v)
                best_category_index = np.argmax(
                    criterion_gains_for_one_feature)
                best_xxx_from_every_feature[col_num] = best_category_index
                best_gain_from_every_feature[col_num] = \
                    criterion_gains_for_one_feature[best_category_index]

            # 7b. Working with numerical features
            else:
                # i) Sort
                sort_indices = np.argsort(feature)
                y_sorted = y[sort_indices]

                # ii) Find uniques in feature
                unique_samples = np.unique(feature)
                n_unique_samples = len(unique_samples)

                print()
                print(f"X[:,{col_num}]: {unique_samples}")
                print(f"     y: {y_sorted}")
                print()

                # iii)
                split_indexes_to_try = select_split_indices(n_unique_samples,
                                                            splits_to_select)

                # iv) Loop through every threshold. For categorical features,
                # thresholds are the categories themselves.
                for split_index in split_indexes_to_try:

                    left, right = np.split(y_sorted, [split_index])

                    criterion = calculate_criterion(
                        self.criteria, left, right, n_classes)
                    weights = np.array([
                        split_index/n_unique_samples,
                        1-split_index/n_unique_samples])
                    weighted_criterion = np.sum(weights * criterion)
                    criterion_gain = criterion_initial - weighted_criterion

                    criterion_gains_for_one_feature.append(criterion_gain)
                    print(left, right)
                    print(f"Criterion gain: {criterion_gain}")

                # v)
                best_split_index = np.argmax(criterion_gains_for_one_feature)
                best_gain_from_every_feature[col_num] = \
                    criterion_gains_for_one_feature[best_split_index]
                best_xxx_from_every_feature[col_num] = \
                    unique_samples[best_split_index]

        print()
        print(
            f"Best criterion gain from every feature:\n{best_gain_from_every_feature}")
        print(
            f"Best split/category from every feature:\n{best_xxx_from_every_feature}")

        # 8. If there are no useful splits, make this node a terminal
        if np.max(best_gain_from_every_feature) < 0.05:
            self.pred = np.argmax(np.bincount(
                y)) if self.criteria == "gini" else np.mean(y)
            self.evaluated = True
            return None, None

        # 9. Find the right question to ask
        self.col = np.argmax(best_gain_from_every_feature)
        self.qn = best_xxx_from_every_feature[self.col]

        print()
        print(f"Best question to ask: Is X[:,{self.col}]<={self.qn}")

        # 10. Split the features into left and right
        # based on the above question.
        best_feature = features[:, self.col]
        if self.col in self.categorical_features:
            left_indices = best_feature == self.qn
            right_indices = best_feature != self.qn
        else:
            left_indices = best_feature <= self.qn
            right_indices = best_feature > self.qn
        left_X_y = features[left_indices], y[left_indices]
        right_X_y = features[right_indices], y[right_indices]

        self.evaluated = True

        return left_X_y, right_X_y

    def __call__(self, x):
        """Calls a node recursively until it hits a prediction"""
        if x.ndim != 1:
            raise ValueError("x must be dim 1")
        if self.pred is not None:
            return self.pred
        elif self.col in self.categorical_features and x[self.col] == self.qn:
            return self.left(x)
        elif self.col not in self.categorical_features and x[self.col] <= self.qn:
            return self.left(x)
        else:
            return self.right(x)

    def __repr__(self):
        if not self.evaluated:
            return f"\n   Id: {self.idx}\n" + \
                f"Depth: {self.depth}\n\n" + \
                f"   Qn: ?\n" + \
                f" Left: ?\n" + \
                f"Right: ?\n\n" + \
                f" Pred: ?\n"
        else:
            return f"\n   Id: {self.idx}\n" + \
                f"Depth: {self.depth}\n\n" + \
                f"   Qn: {self._qn}?\n" + \
                f" Left: {self._left}\n" + \
                f"Right: {self._right}\n\n" + \
                f" Pred: {self._pred}\n"

    @property
    def is_branch(self):
        """The `self.pred` value determines if node is branch"""
        if self.evaluated:
            return self.pred is None
        else:
            raise NotEvaluatedError

    @property
    def is_leaf(self):
        """The `self.pred` value determines if node is leaf"""
        if self.evaluated:
            return not self.is_branch
        else:
            raise NotEvaluatedError

    @property
    def _qn(self):
        """Internal property method used for representing object"""
        if self.qn is None:
            return ""
        else:
            return f"Is X[:,{self.col}] <= {self.qn}?"

    @property
    def _left(self):
        """Internal property method used for representing object"""
        if isinstance(self.pred, int):
            return "-"
        elif self.left is None:
            return "?"
        elif self.left.evaluated is False:
            return "?"
        elif self.left.is_leaf:
            return "(leaf)"
        else:
            return "(branch)"

    @property
    def _right(self):
        """Internal property method used for representing object"""
        if isinstance(self.pred, int):
            return "-"
        elif self.right is None:
            return "?"
        elif self.right.evaluated is False:
            return "?"
        elif self.right.is_leaf:
            return "(leaf)"
        else:
            return "(branch)"

    @property
    def _pred(self):
        """Internal property method used for representing object"""
        if self.pred is None:
            return "-"
        else:
            return self.pred


class ClassificationTreeNode(TreeNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.criteria = "gini"


class RegressionTreeNode(TreeNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.criteria = "mse"

    def __repr__(self):
        return super().__repr__()


class Stack:
    def __init__(self):
        self.data = []

    def push(self, *values):
        self.data.extend(values)

    def pop(self):
        return self.data.pop()

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return self.data.__repr__()

    @property
    def is_not_empty(self):
        return len(self.data) > 0
