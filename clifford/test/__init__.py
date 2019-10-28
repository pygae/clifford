import os
import pytest

def run_all_tests(*args):
    """ Invoke pytest, forwarding options to pytest.main """
    pytest.main([os.path.dirname(__file__)] + list(args))
