import numpy as np
from clifford._numba_utils import generated_jit
import pytest


@generated_jit(cache=True)
def foo(x):
    from clifford.g3 import e3

    def impl(x):
        return (x * e3).value
    return impl


# Make the test fail on a failed cache warning
@pytest.mark.filterwarnings("error")
def test_function_cache():
    from clifford.g3 import e3
    np.testing.assert_array_equal((1.0*e3).value, foo(1.0))
