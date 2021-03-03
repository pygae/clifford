import numpy as np
import pytest

from clifford import Cl, conformalize, _powerset
from clifford._numba_utils import DISABLE_JIT

from . import rng  # noqa: F401

too_slow_without_jit = pytest.mark.skipif(
    DISABLE_JIT, reason="test is too slow without JIT"
)


class TestInitialisation:

    @pytest.mark.parametrize("n", [
        x
        for x in range(2, 7)
    ] + [
        pytest.param(x, marks=too_slow_without_jit)
        for x in range(7, 9)
    ])
    def test_speed(self, n, benchmark):
        def generate_algebra():
            layout = Cl(n)[0]
            layout.gmt_func
            layout.imt_func
            layout.omt_func
            layout.lcmt_func
            layout.adjoint_func
            layout.left_complement_func
            layout.right_complement_func
            layout.dual_func
            layout.vee_func
            layout.inv_func
        benchmark(generate_algebra)

    @too_slow_without_jit
    @pytest.mark.veryslow
    @pytest.mark.parametrize(
        'algebra',
        [Cl(i) for i in [4]] + [conformalize(Cl(3)[0])],
        ids=['Cl(4)', 'conformalize(Cl(3))']
    )
    def test_sparse_multiply(self, algebra, rng):  # noqa: F811
        layout = algebra[0]
        # Make two random multivectors
        a = layout.randomMV(rng=rng)
        b = layout.randomMV(rng=rng)

        # Choose the grades we care about.
        # We skip the cases of:
        #  - all grades
        #  - no grades
        #  - any multiplications including the pseudoscalar
        grades_possibilities = list(_powerset(range(layout.dims)))[1:-1]

        for i, grades_a in enumerate(grades_possibilities):
            sparse_mv_a = sum([a(k) for k in grades_a], layout.MultiVector())
            for j, grades_b in enumerate(grades_possibilities):
                sparse_mv_b = sum([b(k) for k in grades_b], layout.MultiVector())
                # Compute results
                gp = layout.gmt_func_generator(grades_a=grades_a, grades_b=grades_b)
                result_sparse = gp(sparse_mv_a.value, sparse_mv_b.value)
                result_dense = (sparse_mv_a*sparse_mv_b).value
                # Check they are the same
                np.testing.assert_almost_equal(result_sparse, result_dense)
                print(j+i*len(grades_possibilities), len(grades_possibilities)**2)
