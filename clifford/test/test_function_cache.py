import numpy as np
import numba
from numba import generated_jit
import warnings; warnings.simplefilter("error")


@generated_jit(cache=True)
def foo(x):
    from clifford.g3c import e3
    def impl(x):
        return (x * e3).value
    return impl


def test_function_cache():
	from clifford.g3c import e3
	np.testing.assert_array_equal((1.0*e3).value, foo(1.0))
