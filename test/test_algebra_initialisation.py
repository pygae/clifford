from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
from past.builtins import range

from clifford import Cl, randomMV, Frame, get_mult_function, conformalize, grade_obj

from numpy import exp, float64, testing
import unittest
import itertools
import time

from nose.plugins.skip import SkipTest


class InitialisationSpeedTests(unittest.TestCase):

    def test_speed(self):
        algebras = range(2,9)
        print()  # So that the first number is on a new line
        for i in algebras:
            t_start = time.time()
            Cl(i)
            t_end = time.time()
            print(i, t_end - t_start)

    def test_sparse_multiply(self):
        algebras = [Cl(i) for i in [3, 4]] + [conformalize(Cl(3)[0])]
        # For all the algebras we are interested in
        for alg in algebras:
            layout = alg[0]
            # Make two random multivectors
            a = layout.randomMV()
            b = layout.randomMV()
            # Project the multivectors to the grades required
            grades_possibilities = []
            for r in range(1,len(layout.sig)):
                possible_grades = [list(m) for m in list(itertools.combinations(range(len(layout.sig)), r))]
                grades_possibilities += possible_grades
            for i,grades_a in enumerate(grades_possibilities):
                sparse_mv_a = sum([a(k) for k in grades_a])
                for j,grades_b in enumerate(grades_possibilities):
                    sparse_mv_b = sum([b(k) for k in grades_b])
                    # Compute results
                    gp = layout.gmt_func_generator(grades_a=grades_a,grades_b=grades_b)
                    result_sparse = gp(sparse_mv_a.value,sparse_mv_b.value)
                    result_dense = (sparse_mv_a*sparse_mv_b).value
                    # Check they are the same
                    testing.assert_almost_equal(result_sparse, result_dense)
                    print(j+i*len(grades_possibilities),len(grades_possibilities)**2)


if __name__ == '__main__':
    unittest.main()
