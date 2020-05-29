"""
.. currentmodule:: clifford.operator

=============================================
operator functions (:mod:`clifford.operator`)
=============================================

This module exists to enable functional programming via :func:`functools.reduce`.
It can be thought of as equivalent to the builtin :mod:`operator` module, but for the operators from geometric algebra.

    >>> import functools
    >>> import clifford.operator
    >>> from clifford.g3 import *
    >>> Ms = [e1, e1 + e2, e2 + e3]  # list of multivectors
    >>> assert functools.reduce(clifford.operator.op, Ms) == Ms[0] ^ Ms[1] ^ Ms[2]

.. autofunction:: gp

.. autofunction:: op

.. autofunction:: ip

.. versionchanged:: 1.3.0
    These functions used to be in :mod:`clifford`, but have been moved to this
    submodule.

"""
def gp(M, N):
    """
    Geometric product function :math:`MN`, equivalent to ``M * N``.

    M and N must be from the same layout
    """

    return M * N


def ip(M, N):
    r"""
    Hestenes inner product function :math:`M \bullet N`, equivalent to ``M | N``.

    M and N must be from the same layout
    """
    return M | N


def op(M, N):
    r"""
    Outer product function :math:`M \wedge N`, equivalent to ``M ^ N``.

    M and N must be from the same layout
    """
    return M ^ N
