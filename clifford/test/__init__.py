import os
import pytest
import numpy as np


@pytest.fixture
def rng():
    default_test_seed = 1  # the default seed to start pseudo-random tests
    return np.random.default_rng(default_test_seed)


def run_all_tests(*args):
    """ Invoke pytest, forwarding options to pytest.main """
    pytest.main([os.path.dirname(__file__)] + list(args))
