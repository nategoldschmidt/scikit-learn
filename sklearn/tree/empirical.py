from . import _tree

def find_split(X, y, X_argsorted, sample_mask, n_node_samples,
               min_samples_leaf, max_features, criterion, random_state):
    #TODO
    pass


def error_at_node(y, sample_mask, criterion, n_node_samples):
    n_total_samples = y.shape[0]
    criterion.init(y, sample_mask, n_samples, n_total_samples)
    return criterion.eval()


def predict_tree(X, children, feature, threshold, value, n_samples):
    """
    Returns the sum and the number of samples at the resulting leaf
    node.

    """
    #TODO: almost the same as _tree._predict_tree()
    i = 0
    n = X.shape[0]
    node_id = 0
    K = values.shape[1]
    preds = []
    n_samples = []
    for i in range(n):
        node_id = 0
        # While node_id not a leaf
        while children[node_id, 0] != -1 and children[node_id, 1] != -1:
            if X[i, feature[node_id]] <= threshold[node_id]:
                node_id = children[node_id, 0]
            else:
                node_id = children[node_id, 1]
            preds.append(values[node_id])
            n_samples.append(n_samples[node_id])


class EmpiricalCriterion(_tree.Criterion):
    def init_value(self):
        """Return the sum of all responses at this node."""
        pass


class EUCLIDEAN(EmpiricalCriterion):
    pass
