import numpy as np
from clifford._numba_utils import njit
import pytest


@njit(cache=True)
def foo(x, y):
    return (x * y).value


# Make the test fail on a failed cache warning
@pytest.mark.filterwarnings("error")
def test_function_cache():
    from clifford.g3 import e3
    np.testing.assert_array_equal((1.0*e3).value, foo(1.0, e3))
