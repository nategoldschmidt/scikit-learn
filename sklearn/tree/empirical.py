from . import _tree

def find_split(X, y, X_argsorted, sample_mask, n_node_samples,
               min_samples_leaf, max_features, criterion, random_state):
    pass

def error_at_leaf():
    pass

class EmpiricalCriterion(_tree.Criterion):
    pass


class MSE(EmpiricalCriterion):
    pass

class FROBENIUS(EmpiricalCriterion):
    pass

class SSE(EmpiricalCriterion):
    pass

class EUCLIDEAN(EmpiricalCriterion):
    pass
