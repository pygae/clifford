"""
This file tests the behaviour of clifford with complex numbers. The tests
below are based on an informed guess about the correct behavior, and we
should not be afraid of changing them if the things we're testing for turn
out to not be the conventional definitions.
"""

import pytest
import numpy as np
import clifford as cf

from clifford import Cl, conformalize

from . import rng  # noqa: F401


# using fixtures here results in them only being created if needed
@pytest.fixture(scope='module')
def g2():
    return Cl(2)[0]


@pytest.fixture(scope='module')
def g3():
    return Cl(3)[0]


@pytest.fixture(scope='module')
def g4():
    return Cl(4)[0]


@pytest.fixture(scope='module')
def g5():
    return Cl(5)[0]


@pytest.fixture(scope='module')
def g7():
    return Cl(7)[0]


@pytest.fixture(scope='module')
def g3c():
    return conformalize(Cl(3)[0])[0]


@pytest.fixture(scope='module')
def pga():
    from clifford.pga import layout
    return layout


class TestCliffordComplex:
    @pytest.fixture(params=[3, 4, 5, 7, 'g3c', (3, 0, 1)], ids='Cl({})'.format)
    def algebra(self, request, g3, g4, g5, g7, g3c, pga):
        return {3: g3, 4: g4, 5: g5, 7: g7, 'g3c': g3c, (3, 0, 1): pga}[request.param]

    def test_addition(self, algebra, rng):  # noqa: F811
        A = algebra.randomMV(rng=rng)
        B = algebra.randomMV(rng=rng)
        res = (A + 1j*B).value
        res2 = A.value + 1j*B.value
        np.testing.assert_array_equal(res, res2)

    def test_subtraction(self, algebra, rng):  # noqa: F811
        A = algebra.randomMV(rng=rng)
        B = algebra.randomMV(rng=rng)
        res = (A - 1j*B).value
        res2 = A.value - 1j*B.value
        np.testing.assert_array_equal(res, res2)

    @pytest.mark.parametrize('p', [cf.operator.gp, cf.operator.op, cf.operator.ip,
                                   cf.MultiVector.lc, cf.MultiVector.vee])
    def test_prod(self, algebra, p, rng):  # noqa: F811
        A = algebra.randomMV(rng=rng)
        B = algebra.randomMV(rng=rng)
        C = algebra.randomMV(rng=rng)
        D = algebra.randomMV(rng=rng)
        res = (p(A + 1j*B, C + 1j*D)).value
        res2 = p(A, C).value + 1j*p(B, C).value + 1j*p(A, D).value - p(B, D).value
        np.testing.assert_allclose(res, res2)

    def test_reverse(self, algebra, rng):  # noqa: F811
        A = algebra.randomMV(rng=rng)
        B = algebra.randomMV(rng=rng)
        res = (~(A + 1j*B)).value
        res2 = (~A).value + 1j*(~B).value
        np.testing.assert_array_equal(res, res2)

    def test_grade_selection(self, algebra, rng):  # noqa: F811
        A = algebra.randomMV(rng=rng)
        B = algebra.randomMV(rng=rng)
        res = ((A + 1j*B)(2)).value
        res2 = A(2).value + 1j*B(2).value
        np.testing.assert_array_equal(res, res2)

    def test_dual(self, algebra, rng):  # noqa: F811
        A = algebra.randomMV(rng=rng)
        B = algebra.randomMV(rng=rng)
        res = (A + 1j*B).dual().value
        res2 = A.dual().value + 1j*B.dual().value
        np.testing.assert_array_equal(res, res2)

    def test_inverse(self, algebra, rng):  # noqa: F811
        if 0 in algebra.sig:
            pytest.xfail("The inverse in degenerate metrics is known to fail")
        A = algebra.randomMV(rng=rng)
        B = algebra.randomMV(rng=rng)
        original = (A + 1j*B)
        res = algebra.scalar + 0j
        res2 = original*original.inv()
        np.testing.assert_almost_equal(res2.value.real, res.value.real)
        np.testing.assert_almost_equal(res2.value.imag, res.value.imag)
