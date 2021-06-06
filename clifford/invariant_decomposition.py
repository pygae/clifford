"""
.. currentmodule:: clifford.invariant_decomposition

=====================================================
invariant_decomposition (:mod:`clifford.invariant_decomposition`)
=====================================================

.. versionadded:: 1.4.0

This file implements the invariant decomposition (aka bivector split) of bivectors into
mutually commuting orthogonal simple bivectors, based on the method of
M. Roelfs, Spectroscopic and Geometric Algebra Methods for Lattice Gauge Theory, Chapter 6.

The invariant decomposition also enables closed form exponentials and logarithms, and the factorization of
rotors into simple rotors.

Example usage::

>>> B = 1*e12 + 2*e34
>>> bivector_split(B)
[1^e12, 2^e34]

Implemented functions
----------------

.. autofunction:: bivector_split
.. autofunction:: rotor_split
.. autofunction:: exp
.. autofunction:: log

"""
import math
import numpy as np

from ._settings import _eps


def single_split(W_m, li):
    """Helper function to compute the split for a given set of W_m and eigenvalue lambda_i.

    """
    D = sum(W / li**j for j, W in enumerate(W_m[::2]))
    N = sum(W / li**j for j, W in enumerate(W_m[1::2]))
    return N*D.inv()

def bivector_split(B, k=None, roots=False):
    """Bivector split of the bivector B based on the method of M. Roelfs,
    Spectroscopic and Geometric Algebra Methods for Lattice Gauge Theory, Chapter 6.


    Parameters
    ----------
    roots : bool, optional
        If `True`, return the values of the :math:`\lambda_i` in addition to the :math:`b_i`.
        If `False`, return only the :math:`b_i`.
    """
    dim = B.layout.dims
    if k is None:
        k = dim // 2

    W_m = [(B**m)(2*m) / math.factorial(m) for m in range(0, k + 1)]
    W_m_sq = np.array([(W**2).value[0]*(-1)**(k - m) for m, W in enumerate(W_m)])
    ls = np.roots(W_m_sq)

    Bs = []
    for li in sorted(ls, key=lambda li: -np.abs(li))[:-1]:
        # Sort to have the value closest to zero last.
        Bs.append(single_split(W_m, li))
    Bs = Bs + [B - sum(Bs)]
    return (Bs, ls) if roots else Bs

def rotor_split():
    pass

def exp(B):
    Bs, ls = bivector_split(B, roots=True)
    R = 1
    for Bi, li in zip(Bs, ls):
        if isinstance(li, float) and li < 0:
            beta_i = np.sqrt(-li)
            R *= np.cos(beta_i) + (np.sin(beta_i) / beta_i) * Bi
        elif isinstance(li, float) and np.abs(li) < _eps:
            R *= 1 + Bi
        else:
            beta_i = np.sqrt(li)
            R *= np.cosh(beta_i) + (np.sinh(beta_i) / beta_i) * Bi
    return R

def log():
    pass