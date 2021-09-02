"""
.. currentmodule:: clifford.invariant_decomposition

=====================================================
invariant_decomposition (:mod:`clifford.invariant_decomposition`)
=====================================================

.. versionadded:: 1.5.0


This file implements the invariant decomposition (aka bivector split) of bivectors into
mutually commuting orthogonal simple bivectors, based on the method of :cite:`roelfs2021thesis`, chapter 6.

The invariant decomposition enables closed form exponentials and logarithms, and the factorization of
rotors into simple rotors.

Example usage::

    >>> from clifford.g4 import *
    >>> B = 1*e12 + 2*e34
    >>> bivector_split(B)
    [((1+0j)^e12), ((2+0j)^e34)]

Implemented functions
---------------------

.. autofunction:: bivector_split
.. autofunction:: rotor_split
.. autofunction:: exp
.. autofunction:: log


Helper functions
----------------

.. autofunction:: _bivector_split
.. autofunction:: single_split

"""
import math
from functools import reduce

import numpy as np

from ._settings import _eps
from . import _numba_utils


@_numba_utils.njit(cache=True)
def _single_split_even_values(Wm_array, li, r):
    ND = np.zeros((2, Wm_array[0, :].shape[0]), dtype=np.complex_)
    for i in range(0, Wm_array.shape[0]//2+1):
        ND[0, :] += Wm_array[2*i, :] * li**(r - i)
    for i in range(0, Wm_array.shape[0]//2):
        ND[1, :] += Wm_array[2*i+1, :] * li**(r - i - 1)
    return ND


@_numba_utils.njit(cache=True)
def _single_split_odd_values(Wm_array, li, r):
    ND = np.zeros((2, Wm_array[0, :].shape[0]), dtype=np.complex_)
    for i in range(0, Wm_array.shape[0]//2):
        ND[0, :] += Wm_array[2 * i + 1, :] * li ** (r - i)
        ND[1, :] += Wm_array[2*i, :] * li**(r - i)
    return ND


def single_split_even(Wm, li, r):
    """Helper function to compute a single split for a given set of W_m and
    eigenvalue lambda_i, when the total number of terms in the split is even.
    """
    Wm_array = np.array([W.value for W in Wm])
    ND = _single_split_even_values(Wm_array, li, r)
    N = Wm[0].layout.MultiVector(ND[0, :])
    D = Wm[0].layout.MultiVector(ND[1, :])
    return N*D.leftLaInv()


def single_split_odd(Wm, li, r):
    """Helper function to compute a single split for a given set of W_m and
    eigenvalue lambda_i, when the total number of terms in the split is odd.
    """
    Wm_array = np.array([W.value for W in Wm])
    ND = _single_split_odd_values(Wm_array, li, r)
    N = Wm[0].layout.MultiVector(ND[0, :])
    D = Wm[0].layout.MultiVector(ND[1, :])
    return N*D.leftLaInv()


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
    for W in Wm[::-1]:
        if np.linalg.norm(W.value) > _eps:
            break
        else:
            Wm = Wm[:-1]

    k = (len(Wm) - 1)
    r = k // 2
    Wm_sq = np.array([(W ** 2).value[0] * (-1) ** (k - m) for m, W in enumerate(Wm)])
    ls = np.roots(Wm_sq)

    Bs = []
    # Sort to have the value closest to zero last.
    ls_sorted = sorted(ls, key=lambda li: -li)
    # Exclude the smallest value if asked.
    for li in (ls_sorted if return_all else ls_sorted[:-1]):
        if k % 2 == 0:
            Bs.append(single_split_even(Wm, li, r))
        else:
            Bs.append(single_split_odd(Wm, li, r))
    return (Bs, ls_sorted)


def bivector_split(B, k=None, roots=False):
    r"""Bivector split of the bivector B based on the method of :cite:`roelfs2021thesis`, chapter 6.

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

    Rs = [(1 + ti) for ti in Ts]
    Rs = [Ri.normal() if np.isreal((Ri*~Ri).value[0]) else Ri / np.sqrt((Ri*~Ri).value[0]) for Ri in Rs]
    P = reduce(lambda tot, x: tot*x, Rs, 1.0 + 0.0*R)
    Rs = Rs + [R*P.leftLaInv()]
    return (Rs, ls) if roots else Rs


def exp(B):
    Bs, ls = bivector_split(B, roots=True)
    R = B.layout.scalar
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


def log(R):
    Rs, ls = rotor_split(R, roots=True)
    logR = R.layout.MultiVector()
    for Ri, li in zip(Rs, ls):
        if li < 0:
            norm = np.sqrt(- (Ri(2) ** 2).value[0])
            logR += np.arccos(Ri.value[0]) * Ri(2) / norm
        elif np.abs(li) < _eps:
            logR += Ri(2)
        else:
            norm = np.sqrt((Ri(2)**2).value[0])
            logR += np.arccosh(Ri.value[0]) * Ri(2) / norm
    return logR
