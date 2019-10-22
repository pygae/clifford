import itertools
import time

import numpy as np
import pytest

from clifford import Cl, conformalize


class TestInitialisation:

    def test_speed(self):
        algebras = range(2, 9)
        print()  # So that the first number is on a new line
        for i in algebras:
            t_start = time.time()
            Cl(i)
            t_end = time.time()
            print(i, t_end - t_start)

    @pytest.mark.parametrize(
        'algebra',
        [Cl(i) for i in [4]] + [conformalize(Cl(3)[0])],
        ids=['Cl(4)', 'conformalize(Cl(3))']
    )
    def test_sparse_multiply(self, algebra):
        layout = algebra[0]
        # Make two random multivectors
        a = layout.randomMV()
        b = layout.randomMV()
        # Project the multivectors to the grades required
        grades_possibilities = []
        for r in range(1, len(layout.sig)):
            possible_grades = [list(m) for m in list(itertools.combinations(range(len(layout.sig)), r))]
            grades_possibilities += possible_grades
        for i, grades_a in enumerate(grades_possibilities):
            sparse_mv_a = sum([a(k) for k in grades_a])
            for j, grades_b in enumerate(grades_possibilities):
                sparse_mv_b = sum([b(k) for k in grades_b])
                # Compute results
                gp = layout.gmt_func_generator(grades_a=grades_a, grades_b=grades_b)
                result_sparse = gp(sparse_mv_a.value, sparse_mv_b.value)
                result_dense = (sparse_mv_a*sparse_mv_b).value
                # Check they are the same
                np.testing.assert_almost_equal(result_sparse, result_dense)
                print(j+i*len(grades_possibilities), len(grades_possibilities)**2)
