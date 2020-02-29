import numpy as np

__all__ = ["select_features", "select_split_indices", "calculate_criterion",
           "calculate_criterion_initial", "gen"]


def select_features(n_features, method="all"):
    if method == "all":
        return range(n_features)
    elif method == "sqrt":
        n_cols_to_choose = min(1, int(np.ceil(np.sqrt(n_features))))
        return list(np.random.choice(n_features, n_cols_to_choose, replace=False))


def select_split_indices(n_unique_samples, method):
    all_split_indices = range(1, n_unique_samples-1)
    if method == "all":
        return all_split_indices
    elif method == "random":
        return list(np.random.choice(n_features, 1))
    elif isinstance(method, int):
        return list(np.sort(np.random.choice(n_features, method, replace=False)))


def calculate_criterion(criterion, left, right, n_classes):

    if criterion == "gini":
        # `probas`:
        #
        #          probas_left  probas_right
        #  class 0     ...          ...
        #  class 1     ...          ...
        #  class 2     ...          ...
        probas_left = np.bincount(left, minlength=n_classes)/len(left)
        probas_right = np.bincount(
            right, minlength=n_classes)/len(right)
        probas = np.vstack([probas_left, probas_right]).T

        # sum across the classes ie axis 0
        gini = np.sum(probas * (1-probas), axis=0)

        return gini
    else:
        sse_left = ((left-left.mean())**2).sum(keepdims=True)
        sse_right = ((right-right.mean())**2).sum(keepdims=True)
        sse = sse_left + sse_right

        return sse


def calculate_criterion_initial(criterion, y):

    if criterion == "gini":
        proportion = np.bincount(y)/len(y)
        gini_initial = np.sum(proportion*(1-proportion))
        return gini_initial
    else:
        return y.var()


def gen(a, b):
    yield a, b
