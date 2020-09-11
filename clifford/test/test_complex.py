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
def g3c():
    return conformalize(Cl(3)[0])[0]


@pytest.fixture(scope='module')
def pga():
    return Cl(3, 0, 1)[0]


class TestCliffordComplex:
    @pytest.fixture(params=[3, 4, 5, 'g3c', (3, 0, 1)], ids='Cl({})'.format)
    def algebra(self, request, g3, g4, g5, g3c, pga):
        return {3: g3, 4: g4, 5: g5, 'g3c': g3c, (3, 0, 1): pga}[request.param]

    def test_addition(self, algebra):
        A = algebra.randomMV()
        B = algebra.randomMV()
        res = (A + 1j*B).value
        res2 = A.value + 1j*B.value
        np.testing.assert_array_equal(res, res2)

    def test_subtraction(self, algebra):
        A = algebra.randomMV()
        B = algebra.randomMV()
        res = (A - 1j*B).value
        res2 = A.value - 1j*B.value
        np.testing.assert_array_equal(res, res2)

    @pytest.mark.parametrize('p', [cf.operator.gp, cf.operator.op, cf.operator.ip])
    def test_prod(self, algebra, p):
        A = algebra.randomMV()
        B = algebra.randomMV()
        C = algebra.randomMV()
        D = algebra.randomMV()
        res = (p(A + 1j*B, C + 1j*D)).value
        res2 = p(A, C).value + 1j*p(B, C).value + 1j*p(A, D).value - p(B, D).value
        np.testing.assert_allclose(res, res2)

    def test_reverse(self, algebra):
        A = algebra.randomMV()
        B = algebra.randomMV()
        res = (~(A + 1j*B)).value
        res2 = (~A).value + 1j*(~B).value
        np.testing.assert_array_equal(res, res2)

    def test_grade_selection(self, algebra):
        A = algebra.randomMV()
        B = algebra.randomMV()
        res = ((A + 1j*B)(2)).value
        res2 = A(2).value + 1j*B(2).value
        np.testing.assert_array_equal(res, res2)
