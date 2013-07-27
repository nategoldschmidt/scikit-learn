"""Implements the Cheng and Church biclustering algorithm.

Authors : Kemal Eren
License: BSD 3 clause

"""
from abc import ABCMeta

import numpy as np

from sklearn.base import BaseEstimator, BiclusterMixin
from sklearn.externals import six

from sklearn.utils.validation import check_arrays
from sklearn.utils.validation import check_random_state

from .utils import check_array_ndim
from .utils import get_indicators

from ._squared_residue import all_msr


class EmptyBiclusterException(Exception):
    pass


class ChengChurch(six.with_metaclass(ABCMeta, BaseEstimator,
                                     BiclusterMixin)):
    """Algorithm that finds biclusters with small mean squared residue (MSR).

    The residue of an array ``X`` is calculated as ``X -
    X.mean(axis=1)[np.newaxis].T - X.mean(axis=0) + X.mean()``. It
    measures an element's coherence with the overall array, other
    elements in the same row, and other elements in the same column.
    To get the mean squared residue, the residues are squaredd and their
    mean is calculated.

    ChengChurch tries to maximize bicluser size with the constraint
    that its mean squared residue cannot exceed ``max_msr``.

    Parameters
    -----------
    n_clusters : integer, optional, default: 100
        The number of biclusters to find.

    max_msr : float, default: 1.0
        Maximum mean squared residue of a bicluster. Equivalent to
        'delta` in original paper.

    deletion_threshold : float, optional, default: 1.5
        Multiplier for multiple node deletion. Equivalent to `alpha`
        in original paper.

    row_deletion_cutoff : integer, optional, default: 100
        Number of rows at which to switch to single node deletion.

    column_deletion_cutoff : integer, optional, default: 100
        Number of columns at which to switch to single node deletion.

    inverse_rows : bool, optional, default: True
        Whether to add rows with inverse patterns during node addition.

    inverse_columns : bool, optional, default: True
        Whether to add columns with inverse patterns during node addition.

    random_state : int seed, RandomState instance, or None (default)
        A pseudo random number generator used by the K-Means
        initialization.

    Attributes
    ----------
    `rows_` : array-like, shape (n_row_clusters, n_rows)
        Results of the clustering. `rows[i, r]` is True if cluster `i`
        contains row `r`. Available only after calling ``fit()``.

    `columns_` : array-like, shape (n_column_clusters, n_columns)
        Results of the clustering, like `rows`.


    References
    ----------

    - Cheng, Y., & Church, G. M. (2000). `Biclustering of
      expression data
      <ftp://samba.ad.sdsc.edu/pub/sdsc/biology/ISMB00/157.pdf>`__.

    """
    def __init__(self, n_clusters=100, max_msr=1.0, deletion_threshold=1.5,
                 row_deletion_cutoff=100, column_deletion_cutoff=100,
                 inverse_rows=False, inverse_columns=False, random_state=None):
        self.n_clusters = n_clusters
        self.max_msr = max_msr
        self.deletion_threshold = deletion_threshold
        self.row_deletion_cutoff = row_deletion_cutoff
        self.column_deletion_cutoff = column_deletion_cutoff
        self.inverse_rows = inverse_rows
        self.inverse_columns = inverse_columns
        self.random_state = random_state

    def _check_parameters(self):
        if self.n_clusters < 1:
            raise ValueError("'n_clusters' must be > 0, but its value"
                             " is {}".format(self.n_clusters))
        if self.max_msr < 0:
            raise ValueError("'max_msr' must be > 0.0, but its value"
                             " is {}".format(self.max_msr))
        if self.deletion_threshold < 1:
            raise ValueError("'deletion_threshold' must be >= 1.0, but its"
                             " value is {}".format(self.deletion_threshold))
        if self.row_deletion_cutoff < 1:
            raise ValueError("'row_deletion_cutoff' must be >= 1, but its"
                             " value is {}".format(self.row_deletion_cutoff))
        if self.column_deletion_cutoff < 1:
            raise ValueError("'column_deletion_cutoff' must be >= 1, but its"
                             " value is {}".format(
                                 self.column_deletion_cutoff))

    def _all_msr(self, rows, cols, X):
        if not rows.size or not cols.size:
            raise EmptyBiclusterException()
        return all_msr(rows.ravel(), cols, X)

    def _row_msr(self, rows, cols, X, inverse=False):
        # TODO: not necessary for rows already in bicluster
        if not rows.size or not cols.size:
            raise EmptyBiclusterException()
        row_mean = X[:, cols].mean(axis=1)[np.newaxis].T
        col_mean = X[rows, cols].mean(axis=0)
        if inverse:
            arr = -X[:, cols] + row_mean - col_mean + X[rows, cols].mean()
        else:
            arr = X[:, cols] - row_mean - col_mean + X[rows, cols].mean()
        return np.power(arr, 2).mean(axis=1)

    def _col_msr(self, rows, cols, X, inverse=False):
        # TODO: not necessary for columns already in bicluster
        if not rows.size or not cols.size:
            raise EmptyBiclusterException()
        rows_t = rows
        rows = rows_t.ravel()
        row_mean = X[rows_t, cols].mean(axis=1)[np.newaxis].T
        col_mean = X[rows, :].mean(axis=0)
        if inverse:
            arr = -X[rows, :] + row_mean - col_mean + X[rows_t, cols].mean()
        else:
            arr = X[rows, :] - row_mean - col_mean + X[rows_t, cols].mean()
        return np.power(arr, 2).mean(axis=0)

    def _single_node_deletion(self, rows, cols, X):
        msr, row_msr, col_msr = self._all_msr(rows, cols, X)
        while msr > self.max_msr:
            n_rows, n_cols = len(rows), len(cols)
            row_id = np.argmax(row_msr)
            col_id = np.argmax(col_msr)
            if row_msr[row_id] > col_msr[col_id]:
                rows = rows.ravel()
                rows = np.setdiff1d(rows, [rows[row_id]])[np.newaxis].T
            else:
                cols = np.setdiff1d(cols, [cols[col_id]])
            msr, row_msr, col_msr = self._all_msr(rows, cols, X)
        return rows, cols

    def _multiple_node_deletion(self, rows, cols, X):
        msr, row_msr, col_msr = self._all_msr(rows, cols, X)
        while msr > self.max_msr:
            n_rows, n_cols = len(rows), len(cols)
            if n_rows >= self.row_deletion_cutoff:
                to_remove = row_msr > (self.deletion_threshold * msr)
                rows = rows.ravel()
                rows = np.setdiff1d(rows, rows[to_remove])[np.newaxis].T

            msr, row_msr, col_msr = self._all_msr(rows, cols, X)
            if n_cols >= self.column_deletion_cutoff:
                to_remove = col_msr > (self.deletion_threshold * msr)
                cols = np.setdiff1d(cols, cols[to_remove])

            if n_rows == len(rows) and n_cols == len(cols):
                break
            msr, row_msr, col_msr = self._all_msr(rows, cols, X)
        return rows, cols

    def _node_addition(self, rows, cols, X):
        while True:
            n_rows, n_cols = len(rows), len(cols)
            msr, _, _ = self._all_msr(rows, cols, X)
            col_msr = self._col_msr(rows, cols, X)
            to_add = np.nonzero(col_msr < msr)[0]
            old_cols = cols.copy()  # save for inverse
            cols = np.union1d(cols, to_add)

            if self.inverse_columns:
                col_msr = self._col_msr(rows, old_cols, X,
                                        inverse=True).mean(axis=1)
                to_add = np.nonzero(col_msr < msr)[0]
                cols = np.union1d(cols, to_add)

            msr, _, _ = self._all_msr(rows, cols, X)
            row_msr = self._row_msr(rows, cols, X)
            to_add = np.nonzero(row_msr < msr)[0]
            old_rows = rows.copy()  # save for inverse
            rows = np.union1d(rows.ravel(), to_add)[np.newaxis].T

            if self.inverse_rows:
                row_msr = self._row_msr(old_rows, cols, X,
                                        inverse=True).mean(axis=1)
                to_add = np.nonzero(row_msr < msr)[0]
                rows = np.union1d(rows.ravel(), to_add)[np.newaxis].T

            if n_rows == len(rows) and n_cols == len(cols):
                break
        return rows, cols

    def _mask(self, X, rows, cols, generator, minval, maxval):
        mask_vals = generator.uniform(minval, maxval, (len(rows), len(cols)))
        X[rows, cols] = mask_vals

    def fit(self, X):
        X = X.copy()  # need to modify it in-place
        self._check_parameters()
        X, = check_arrays(X, dtype=np.float64)
        check_array_ndim(X)
        minval, maxval = X.min(), X.max()
        n_rows, n_cols = X.shape

        generator = check_random_state(self.random_state)
        results = []

        for i in range(self.n_clusters):
            rows = np.arange(n_rows, dtype=np.int)[np.newaxis].T
            cols = np.arange(n_cols, dtype=np.int)
            rows, cols = self._multiple_node_deletion(rows, cols, X)
            rows, cols = self._single_node_deletion(rows, cols, X)
            rows, cols = self._node_addition(rows, cols, X)
            self._mask(X, rows, cols, generator, minval, maxval)
            results.append((rows.ravel(), cols))

        indicators = (get_indicators(r, c, X.shape) for r, c in results)
        rows, cols = zip(*indicators)
        self.rows_ = np.vstack(rows)
        self.columns_ = np.vstack(cols)
