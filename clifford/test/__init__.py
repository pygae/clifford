import os
import pytest

default_test_seed = 1  # the default seed to start pseudo-random tests


def run_all_tests(*args):
    """ Invoke pytest, forwarding options to pytest.main """
    pytest.main([os.path.dirname(__file__)] + list(args))
