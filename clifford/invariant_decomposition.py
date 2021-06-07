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
from functools import reduce

import numpy as np

from ._settings import _eps


def single_split(Wm, li):
    """Helper function to compute the split for a given set of W_m and eigenvalue lambda_i.

    """
    D = sum(W / li**j for j, W in enumerate(Wm[::2]))
    N = sum(W / li**j for j, W in enumerate(Wm[1::2]))
    return N*D.inv()

def _bivector_split(Wm, return_all=True):
    """Internal helper function to perform the decomposition, given a set of Wm.

    Parameters
    ----------
    return_all : bool, optional
        If `True`, returns all the :math:`b_i`.
        If `False`, return all :math:`b_i` except for the one with the smallest magnitude.
    """
    # The effective value of k is determined by the largest non-zero W.
    # remove the highest grade zeros to prevent meaningless lambda_i = 0 values.
    for W in reversed(Wm):
        if np.linalg.norm(W.value) > _eps:
            break
        else:
            Wm = Wm[:-1]

    k = (len(Wm) - 1)
    Wm_sq = np.array([(W ** 2).value[0] * (-1) ** (k - m) for m, W in enumerate(Wm)])
    ls = np.roots(Wm_sq)

    Bs = []
    # Sort to have the value closest to zero last.
    ls_sorted = sorted(ls, key=lambda li: -np.abs(li))
    # Exclude the smallest value if asked.
    ls_sorted = ls_sorted if return_all else ls_sorted[:-1]

    for li in ls_sorted:
        Bs.append(single_split(Wm, li))
    return (Bs, ls)

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

    Wm = [(B**m)(2*m) / math.factorial(m) for m in range(0, k + 1)]
    Bs, ls = _bivector_split(Wm, return_all=False)
    Bs = Bs + [B - sum(Bs)]
    return (Bs, ls) if roots else Bs

def rotor_split(R, k=None, roots=False):
    dim = R.layout.dims
    if k is None:
        k = dim // 2

    Wm = [R(2 * m) for m in range(0, k + 1)]
    Ts, ls = _bivector_split(Wm, return_all=False)

    Rs = [(R(0) + R(0) * ti) for ti in Ts]
    Rs = [Ri.normal() if np.isreal((Ri*(~Ri)).value[0]) else Ri / np.sqrt((Ri*(~Ri)).value[0]) for Ri in Rs]
    P = reduce(lambda tot, x: tot*x, Rs, 1)
    Rs = Rs + [R/P]
    return (Rs, ls) if roots else Rs

def exp(B):
    Bs, ls = bivector_split(B, roots=True)
    R = 1
    for Bi, li in zip(Bs, ls):
        if np.isreal(li) and li < 0:
            beta_i = np.sqrt(-li)
            R *= np.cos(beta_i) + (np.sin(beta_i) / beta_i) * Bi
        elif np.isreal(li) and np.abs(li) < _eps:
            R *= 1 + Bi
        else:
            beta_i = np.sqrt(li)
            R *= np.cosh(beta_i) + (np.sinh(beta_i) / beta_i) * Bi
    return R

def log():
    pass
