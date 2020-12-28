"""
.. currentmodule:: clifford.taylor_expansions

=====================================================
taylor_expansions (:mod:`clifford.taylor_expansions`)
=====================================================

.. versionadded:: 1.4.0

This file implements various Taylor expansions for useful functions of multivectors.
For some algebra signatures there may exist closed forms of these functions which would likely be faster
and more accurate. Nonetheless, having pre-written taylor expansions for the general case is useful.

.. note::
    Many of these functions are also exposed as :class:`~clifford.MultiVector` methods,
    such as :meth:`clifford.MultiVector.sin`. This means that ``mv.sin()`` or even ``np.sin(mv)`` can be used
    as a convenient interface to functions in this module, without having to import it directly.

    For example::

    >>> from clifford.g3 import *
    >>> import numpy as np
    >>> np.sin(np.pi*e12/4)
    (0.86867^e12)

Implemented functions
----------------

.. autofunction:: exp
.. autofunction:: sin
.. autofunction:: cos
.. autofunction:: tan
.. autofunction:: sinh
.. autofunction:: cosh
.. autofunction:: tanh

"""
import math
import numpy as np

from . import _numba_utils
from . import _settings


@_numba_utils.njit
def exp(x, max_order=15):
    """
    This implements the series expansion of :math:`\exp x` where :math:`x` is a multivector
    The parameter `max_order` is the maximum order of the taylor series to use
    """

    result = 1.0 + 0.0*x
    if max_order == 0:
        return result

    # scale by power of 2 so that its norm is < 1
    max_val = int(np.max(np.abs(x.value)))
    scale = 1
    if max_val > 1:
        max_val <<= 1
    while max_val:
        max_val >>= 1
        scale <<= 1

    scaled = x * (1.0 / scale)

    # taylor approximation
    tmp = 1.0 + 0.0*x
    for i in range(1, max_order):
        if np.any(np.abs(tmp.value) > _settings._eps):
            tmp = tmp*scaled * (1.0 / i)
            result = result + tmp
        else:
            break

    # undo scaling
    while scale > 1:
        result = result*result
        scale >>= 1
    return result


@_numba_utils.njit
def sin(X, max_order=30):
    """
    A taylor series expansion for sin
    The parameter `max_order` is the maximum order of the taylor series to use
    """
    op = +X
    X2 = X*X
    X2np1 = X
    for n in range(1, max_order):
        X2np1 = X2np1 * X2
        op = op + ((-1) ** (n) / math.gamma(2 * n + 2)) * X2np1
    return op


@_numba_utils.njit
def cos(X, max_order=30):
    """
    A taylor series expansion for cos
    The parameter `max_order` is the maximum order of the taylor series to use
    """
    op = 1 + 0*X
    X2 = X * X
    X2n = 1 + 0*X
    for n in range(1, max_order):
        X2n = X2n*X2
        op = op + ((-1) ** (n) / math.gamma(2 * n + 1)) * X2n
    return op


def tan(X, max_order=30):
    """
    The tan function as the ratio of sin and cos
    The parameter `max_order` is the maximum order of the taylor series to use

    .. note::
        It would probably be better to implement this as its own taylor series. This function
        is not JITed as currently we do not overload the truediv operator for multivectors.
    """
    return sin(X, max_order) / cos(X, max_order)


@_numba_utils.njit
def sinh(X, max_order=30):
    """
    A taylor series expansion for sinh
    The parameter `max_order` is the maximum order of the taylor series to use
    """
    op = +X
    X2 = X * X
    X2np1 = X
    for n in range(1, max_order):
        X2np1 = X2np1 * X2
        op = op + (1 / math.gamma(2 * n + 2)) * X2np1
    return op


@_numba_utils.njit
def cosh(X, max_order=30):
    """
    A taylor series expansion for cosh
    The parameter `max_order` is the maximum order of the taylor series to use
    """
    op = 1 + 0 * X
    X2 = X * X
    X2n = 1 + 0 * X
    for n in range(1, max_order):
        X2n = X2n * X2
        op = op + (1 / math.gamma(2 * n + 1)) * X2n
    return op


def tanh(X, max_order=30):
    """
    The tanh function as the ratio of sinh and cosh
    The parameter `max_order` is the maximum order of the taylor series to use

    .. note::
        It would probably be better to implement this as its own taylor series. This function
        is not JITed as currently we do not overload the truediv operator for multivectors.
    """
    return sinh(X, max_order) / cosh(X, max_order)
