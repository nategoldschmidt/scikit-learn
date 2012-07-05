from . import _tree


class EmpiricalCriterion(_tree.Criterion):
    def init_value(self):
        """Return the sum of all responses at this node."""
        pass


class EUCLIDEAN(EmpiricalCriterion):
    pass


def error_at_node(y, sample_mask, criterion, n_node_samples):
    n_total_samples = y.shape[0]
    criterion.init(y, sample_mask, n_samples, n_total_samples)
    return criterion.eval()


def find_best_split(X,
                    y
                    X_argsorted,
                    sample_mask,
                    n_samples,
                    min_leaf,
                    max_features,
                    criterion,
                    random_state):
    # Variables declarations
    n_total_samples = X.shape[0]
    n_features = X.shape[1]
    i = -1
    a = -1
    b = -1
    best_i = -1

    feature_idx = -1
    n_left = 0
    best_error = np.inf
    best_t = np.inf

    # Compute the initial criterion value in the node
    criterion.init(y, sample_mask, n_samples, n_total_samples)
    initial_error = criterion.eval()

    if initial_error == 0:  # break early if the node is pure
        return best_i, best_t, initial_error, initial_error

    best_error = initial_error

    # Features to consider
    features = np.arange(n_features, dtype=np.int32)
    if max_features < 0 or max_features >= n_features:
        max_features = n_features
    else:
        features = random_state.permutation(features)[:max_features]

    # Look for the best split
    for feature_idx in range(max_features):
        i = features[feature_idx]
        # Get i-th col of X and X_sorted
        X_i = (<DTYPE_t *>X.data) + X_stride * i
        X_argsorted_i = (<int *>X_argsorted.data) + X_argsorted_stride * i

        # Reset the criterion for this feature
        criterion.reset()

        # Index of smallest sample in X_argsorted_i that is in the sample mask
        a = 0
        while sample_mask_ptr[X_argsorted_i[a]] == 0:
            a = a + 1

        # Consider splits between two consecutive samples
        while True:
            # Find the following larger sample
            b = smallest_sample_larger_than(a, X_i, X_argsorted_i,
                                            sample_mask_ptr, n_total_samples)
            if b == -1:
                break

            # Better split than the best so far?
            n_left = criterion.update(a, b, y_ptr, X_argsorted_i, sample_mask_ptr)

            # Only consider splits that respect min_leaf
            if n_left < min_leaf or (n_samples - n_left) < min_leaf:
                a = b
                continue

            error = criterion.eval()

            if error < best_error:
                t = X_i[X_argsorted_i[a]] + \
                    ((X_i[X_argsorted_i[b]] - X_i[X_argsorted_i[a]]) / 2.0)
                if t == X_i[X_argsorted_i[b]]:
                    t = X_i[X_argsorted_i[a]]
                best_i = i
                best_t = t
                best_error = error

            # Proceed to the next interval
            a = b

    return best_i, best_t, best_error, initial_error
