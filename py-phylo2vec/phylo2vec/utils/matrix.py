"""Phylo2Vec matrix manipulation functions."""

import numpy as np

from phylo2vec import _phylo2vec_core as core


def check_matrix(m: np.ndarray) -> None:
    """Input validation of a Phylo2Vec matrix

    The input is checked to satisfy the Phylo2Vec constraints

    Parameters
    ----------
    m : numpy.ndarray
        Phylo2Vec matrix
    """
    core.check_m(m)


def sample_matrix(n_leaves: int, ordered: bool = False) -> np.ndarray:
    """Sample a random tree with branch lengths via Phylo2Vec, in matrix form.

    Parameters
    ----------
    n_leaves : int
        Number of leaves
    ordered : bool, optional
        If True, sample an ordered tree, by default False

    Returns
    -------
    numpy.ndarray
        Phylo2Vec matrix
        Dimensions (n_leaves, 3)
        1st column: Phylo2Vec vector
        2nd and 3rd columns: branch lengths of cherry [i] in the ancestry matrix
    """

    return core.sample_matrix(n_leaves, ordered)


def precision(m: np.ndarray) -> np.ndarray:
    """Convert a Phylo2Vec matrix to a specified precision.

    Parameters
    ----------
    m : numpy.ndarray
        Phylo2Vec matrix
    precision : int, optional
        Number of decimal places, by default 6

    Returns
    -------
    numpy.ndarray
        Phylo2Vec matrix with specified precision
    """
    k = m.shape[0]
    n_leaves = k + 1
    p = core.precision(m)
    a = p[:n_leaves, :n_leaves]
    b = p[:n_leaves, n_leaves:]
    c = p[n_leaves:, n_leaves:]

    return a - b @ np.linalg.solve(c, b.T)
