from . import _tree


class EmpiricalCriterion(_tree.Criterion):
    def init_value(self):
        """Return the sum of all responses at this node."""
        pass


class EUCLIDEAN(EmpiricalCriterion):
    pass
