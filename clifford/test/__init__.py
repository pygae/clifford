import os
import pytest
import numpy as np

from clifford._numba_utils import DISABLE_JIT


@pytest.fixture
def rng():
    default_test_seed = 1  # the default seed to start pseudo-random tests
    return np.random.default_rng(default_test_seed)


too_slow_without_jit = pytest.mark.skipif(
    DISABLE_JIT, reason="test is too slow without JIT"
)


def run_all_tests(*args):
    """ Invoke pytest, forwarding options to pytest.main """
    pytest.main([os.path.dirname(__file__)] + list(args))
