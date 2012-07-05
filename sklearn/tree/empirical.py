import sys
import _tree
import numpy as np

class EmpiricalCriterion(_tree.Criterion):

    def __init__(self):
        self.responses = None
        self.sample_mask = None
        self.responses_l = None
        self.responses_r = None
        self.n_samples = 0


    def init(self, y, sample_mask, n_samples,
                   n_total_samples):
        """Initialise the criterion class for new split point."""
        assert len(y) == n_total_samples
        assert [t for t in sample_mask].count(True) == n_samples
        self.responses = y
        self.sample_mask = sample_mask
        self.n_samples = n_samples
        self.reset()


    def update(self, a, b, y, X_argsorted_i,
                    sample_mask):
        """Update the criteria for each value in interval [a,b) (where a and b
           are indices in `X_argsorted_i`)."""
        for i in range(len(y)):
            s = X_argsorted_i[i]
            if not sample_mask[s]:
                continue
            if i < b:
                self.responses_l.append(y[s])
            else:
                self.responses_r.append(y[s])
        return len(self.responses_l)


    def reset(self):
        """Reset the criterion for a new feature index."""
        self.responses_l = []
        self.responses_r = []
        for r in self.responses:
            self.responses_r.append(r)


    def init_value(self):
        """Return all responses at this node."""
        return [r for r, m in zip(self.responses, self.sample_mask) if m]


class Euclidean(EmpiricalCriterion):
    def eval(self):
        """Evaluate the criteria (aka the split error)."""
        sum_r = sum(self.responses_r)
        n_r = len(self.responses_r)
        mean_r = np.ravel(sum_r / n_r)
        rdiff = n_r / self.n_samples * sum([np.linalg.norm(np.ravel(r) - mean_r) for r in self.responses_r])

        if len(self.responses_l) == 0:
            ldiff = 0
        else:
            sum_l = sum(self.responses_l)
            n_l = len(self.responses_l)
            mean_l = np.ravel(sum_l / n_l)
            ldiff = n_l / self.n_samples * sum([np.linalg.norm(np.ravel(r) - mean_l) for r in self.responses_l])

        return ldiff + rdiff


def error_at_leaf(y, sample_mask, criterion, n_node_samples):
    n_total_samples = y.shape[0]
    criterion.init(y, sample_mask, n_node_samples, n_total_samples)
    return criterion.eval()


def smallest_sample_larger_than(sample_idx, X_i,
                                X_argsorted_i, sample_mask,
                                n_total_samples):
    threshold = -sys.float_info.max

    if sample_idx > -1:
        threshold = X_i[X_argsorted_i[sample_idx]]

    for idx in range(sample_idx + 1, n_total_samples):
        j = X_argsorted_i[idx]

        if sample_mask[j] == 0:
            continue

        if X_i[j] > threshold + 1.e-7:
            return idx

    return -1



def find_best_split(X,
                    y,
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
        X_i = X[:,i]
        X_argsorted_i = X_argsorted[:,i]

        # Reset the criterion for this feature
        criterion.reset()

        # Index of smallest sample in X_argsorted_i that is in the sample mask
        a = 0
        while sample_mask[X_argsorted_i[a]] == 0:
            a = a + 1

        # Consider splits between two consecutive samples
        while True:
            # Find the following larger sample
            b = smallest_sample_larger_than(a, X_i, X_argsorted_i,
                                            sample_mask, n_total_samples)
            if b == -1:
                break

            # Better split than the best so far?
            n_left = criterion.update(a, b, y, X_argsorted_i, sample_mask)

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
