import numpy as np
from .data_structures import Stack


def plant_tree(Node, X, y, categorical,
               max_depth,
               features_to_select,
               splits_to_select):
    """
    This tree builder controls how the decision tree
    is constructed using the params. This is done by passing
    these params into the `TreeNode.split()` method.

    Params
    ------
    Node: class
        ClassificationTreeNode or RegressionTreeNode
    max_depth: int
        Depth of tree to grow. 0 indicates there's only the
        root node. -1 indicates growing tree uncontrollably.
    features_to_select: str
        Features to select at every split. "all" for all features,
        "sqrt" for sqrt(p) features where p is the no. of features available
        for that split.
    splits_to_select: str or int
        Only for numerical features. "all" to use all unique values,
        "random" to use only one of the unique values, and
        an integer k to use k of the unique values.
    """
    stack = Stack()

    idx = 0
    root = Node(data=(X, y), categorical_features=categorical, idx=idx, depth=0)
    stack.push(root)

    while stack.is_not_empty:

        node = stack.pop()
        data_left, data_right = node.split(max_depth, features_to_select,
                                           splits_to_select)

        if node.is_branch:

            idx += 1
            node_left = Node(data=data_left, categorical_features=categorical,
                             idx=idx, depth=node.depth+1)
            idx += 1
            node_right = Node(data=data_right, categorical_features=categorical,
                              idx=idx, depth=node.depth+1)

            node.left = node_left
            node.right = node_right

            # TODO ideally only push one node
            stack.push(node_left, node_right)

    return root


def plant_tree_average(Node, y):
    node = Node(data=(0,0), idx=0, depth=0)
    node.pred = np.mean(y)
    node.evaluated = True
    return node
