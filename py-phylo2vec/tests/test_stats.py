"""Tests for metrics."""

import numpy as np
import pytest

from ete3 import Tree
from scipy import sparse

from phylo2vec.base.newick import to_newick
from phylo2vec.stats.nodewise import (
    cophenetic_distances,
    cov,
    incidence,
    pairwise_distances,
    precision,
)
from phylo2vec.utils.matrix import sample_matrix
from phylo2vec.utils.vector import sample_vector
from .config import MIN_N_LEAVES, N_REPEATS


MAX_N_LEAVES_STATS = 50


def _test_cophenetic(n_leaves, sample_fn):
    """Helper function to test cophenetic distances using a sample function."""

    def coph_ete3(tr, n_leaves):
        dmat = np.zeros((n_leaves, n_leaves))

        for i in range(n_leaves):
            for j in range(i):
                dist = tr.get_distance(f"{i}", f"{j}", topology_only=False)
                dmat[i, j] = dist
                dmat[j, i] = dist

        return dmat

    for _ in range(N_REPEATS):
        vector_or_matrix = sample_fn(n_leaves)

        # tree with all branch lengths = 1
        tr = Tree(to_newick(vector_or_matrix))

        # Our distance matrix
        dmat_p2v = cophenetic_distances(vector_or_matrix)

        # ete3 distance matrix
        dmat_ete3 = coph_ete3(tr, n_leaves)

        assert np.allclose(dmat_p2v, dmat_ete3)


@pytest.mark.parametrize("n_leaves", range(MIN_N_LEAVES, MAX_N_LEAVES_STATS + 1))
def test_cophenetic_vector(n_leaves):
    """Test that v to newick to converted_v leads to v == converted_v

    Parameters
    ----------
    n_leaves : int
        Number of leaves
    """

    _test_cophenetic(n_leaves, sample_vector)


@pytest.mark.parametrize("n_leaves", range(MIN_N_LEAVES, MAX_N_LEAVES_STATS + 1))
def test_cophenetic_matrix(n_leaves):
    """Test that v to newick to converted_v leads to v == converted_v

    Parameters
    ----------
    n_leaves : int
        Number of leaves
    """

    _test_cophenetic(n_leaves, sample_matrix)


@pytest.mark.parametrize("n_leaves", [MIN_N_LEAVES, MAX_N_LEAVES_STATS + 1])
def test_pairwise_distances_cophenetic(n_leaves):
    """Test the `pairwise_distances` function with cophenetic distances.

    Parameters
    ----------
    n_leaves : int
        Number of leaves
    """
    # Test with a vector
    vector = sample_vector(n_leaves)
    dmat_vector = pairwise_distances(vector, metric="cophenetic")
    dmat_vector2 = cophenetic_distances(vector)
    assert np.array_equal(dmat_vector, dmat_vector2)

    # Test with a matrix
    matrix = sample_matrix(n_leaves)
    dmat_matrix = pairwise_distances(matrix, metric="cophenetic")
    dmat_matrix2 = cophenetic_distances(matrix)
    assert np.array_equal(dmat_matrix, dmat_matrix2)


def _test_cov_and_precision(n_leaves, sample_fn):
    for _ in range(N_REPEATS):
        vector_or_matrix = sample_fn(n_leaves)
        cov_matrix = cov(vector_or_matrix)
        precision_matrix = precision(vector_or_matrix)
        identity = np.eye(n_leaves, dtype=cov_matrix.dtype)
        # Note: this test might fail for larger n_leaves due to numerical precision issues
        assert np.allclose(cov_matrix @ precision_matrix, identity)
        assert np.allclose(precision_matrix @ cov_matrix, identity)


@pytest.mark.parametrize("n_leaves", range(MIN_N_LEAVES, MAX_N_LEAVES_STATS + 1))
def test_cov_vector(n_leaves):
    """Test covariance matrix for vector input.

    Parameters
    ----------
    n_leaves : int
        Number of leaves
    """
    _test_cov_and_precision(n_leaves, sample_vector)


@pytest.mark.parametrize("n_leaves", range(MIN_N_LEAVES, MAX_N_LEAVES_STATS + 1))
def test_cov_matrix(n_leaves):
    """Test covariance matrix for matrix input.

    Parameters
    ----------
    n_leaves : int
        Number of leaves
    """
    _test_cov_and_precision(n_leaves, sample_matrix)


@pytest.mark.parametrize(
    "v, d",
    [
        (np.array([0]), np.array([[1, 0], [0, 1], [-1, -1]])),
        (
            np.array([0, 1]),
            np.array(
                [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, -1, -1, 1],
                    [-1, 0, 0, -1],
                ]
            ),
        ),
        (
            np.array([0, 1, 2]),
            np.array(
                [
                    [1, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0],
                    [0, 0, -1, -1, 1, 0],
                    [0, -1, 0, 0, -1, 1],
                    [-1, 0, 0, 0, 0, -1],
                ]
            ),
        ),
    ],
)
def test_incidence(v, d):
    """Test incidence matrix for different formats.

    Parameters
    ----------
    v : np.ndarray
        Input vector.
    d : np.ndarray
        Expected incidence matrix.
    """
    format_fns = {
        "coo": sparse.coo_matrix,
        "csr": sparse.csr_matrix,
        "csc": sparse.csc_matrix,
        "dense": lambda x: x,
    }
    for f, func in format_fns.items():
        inc = incidence(v, format=f)
        if f == "dense":
            assert np.array_equal(inc, d)
        elif f == "coo":
            data, rows, cols = inc
            assert np.array_equal(func((data, (rows, cols))).toarray(), d)
        else:
            assert np.array_equal(func(inc).toarray(), d)
    # assert np.array_equal(incidence(v, format="dense"), d)


if __name__ == "__main__":
    pytest.main()
