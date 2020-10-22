
import math

import numpy as np

from . import _numba_utils
from . import _settings


@_numba_utils.njit
def general_exp(x, max_order=15):
    """
    This implements the series expansion of e**mv where mv is a multivector
    The parameter order is the maximum order of the taylor series to use
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
def general_sin(X, max_order=30):
    """
    A taylor series expansion for sin
    """
    op = +X
    X2 = X*X
    X2np1 = X
    for n in range(1, max_order):
        X2np1 = X2np1 * X2
        op = op + ((-1) ** (n) / math.gamma(2 * n + 2)) * X2np1
    return op


@_numba_utils.njit
def general_cos(X, max_order=30):
    """
    A taylor series expansion for cos
    """
    op = 1 + 0*X
    X2 = X * X
    X2n = 1 + 0*X
    for n in range(1, max_order):
        X2n = X2n*X2
        op = op + ((-1) ** (n) / math.gamma(2 * n + 1)) * X2n
    return op


def general_tan(X, max_order=30):
    """
    The tan function as the ratio of sin and cos
    Note. It would probably be better to implement this as its own taylor series.
    """
    return general_sin(X, max_order) / general_cos(X, max_order)


@_numba_utils.njit
def general_sinh(X, max_order=30):
    """
    A taylor series expansion for sinh
    """
    op = +X
    X2 = X * X
    X2np1 = X
    for n in range(1, max_order):
        X2np1 = X2np1 * X2
        op = op + (1 / math.gamma(2 * n + 2)) * X2np1
    return op


@_numba_utils.njit
def general_cosh(X, max_order=30):
    """
    A taylor series expansion for cosh
    """
    op = 1 + 0 * X
    X2 = X * X
    X2n = 1 + 0 * X
    for n in range(1, max_order):
        X2n = X2n * X2
        op = op + (1 / math.gamma(2 * n + 1)) * X2n
    return op


def general_tanh(X, max_order=30):
    """
    The tanh function as the ratio of sinh and cosh
    Note. It would probably be better to implement this as its own taylor series.
    """
    return general_sinh(X, max_order) / general_cosh(X, max_order)
