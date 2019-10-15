from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
from past.builtins import range

from clifford import Cl, randomMV, Frame, get_mult_function, conformalize, grade_obj
from clifford.tools import orthoFrames2Verser as of2v

import numpy as np
from numpy import exp, float64, testing
import unittest
import itertools
import time

from nose.plugins.skip import SkipTest


class InitialisationSpeedTests(unittest.TestCase):

    def test_speed(self):
        algebras = [2,3,4,5,6,7,8]
        t_start = time.time()
        for i in algebras:
            Cl(i)
            print(time.time() - t_start)

if __name__ == '__main__':
    unittest.main()
