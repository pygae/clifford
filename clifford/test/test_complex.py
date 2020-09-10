
import pytest
import numpy as np


from clifford import Cl, randomMV, Frame, \
    conformalize, grade_obj, MultiVector

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
def g4_1():
    return Cl(4, 1)[0]


@pytest.fixture(scope='module')
def g3c(g3):
    return conformalize(g3)[0]


class TestCliffordComplex:
    @pytest.fixture(params=[3, 4, 5], ids='Cl({})'.format)
    def algebra(self, request, g3, g4, g5):
        return {3: g3, 4: g4, 5: g5}[request.param]

    def test_addition(self, algebra):
        A = algebra.randomMV()
        B = algebra.randomMV()
        res = (A + 1j*B).value
        res2 = A.value + 1j*B.value
        np.testing.assert_allclose(res, res2)

    def test_subtraction(self, algebra):
        A = algebra.randomMV()
        B = algebra.randomMV()
        res = (A - 1j*B).value
        res2 = A.value - 1j*B.value
        np.testing.assert_allclose(res, res2)

    def test_gp(self, algebra):
        A = algebra.randomMV()
        B = algebra.randomMV()
        C = algebra.randomMV()
        D = algebra.randomMV()
        res = ((A + 1j*B)*(C + 1j*D)).value
        res2 = (A*C).value + 1j*(B*C).value + 1j*(A*D).value - (B*D).value 
        np.testing.assert_allclose(res, res2)

    def test_op(self, algebra):
        A = algebra.randomMV()
        B = algebra.randomMV()
        C = algebra.randomMV()
        D = algebra.randomMV()
        res = ((A + 1j*B)^(C + 1j*D)).value
        res2 = (A^C).value + 1j*(B^C).value + 1j*(A^D).value - (B^D).value 
        np.testing.assert_allclose(res, res2)

    def test_ip(self, algebra):
        A = algebra.randomMV()
        B = algebra.randomMV()
        C = algebra.randomMV()
        D = algebra.randomMV()
        res = ((A + 1j*B)|(C + 1j*D)).value
        res2 = (A|C).value + 1j*(B|C).value + 1j*(A|D).value - (B|D).value 
        np.testing.assert_allclose(res, res2)

    def test_reverse(self, algebra):
        A = algebra.randomMV()
        B = algebra.randomMV()
        res = (~(A + 1j*B)).value
        res2 = (~A).value + 1j*(~B).value
        np.testing.assert_allclose(res, res2)

    def test_grade_selection(self, algebra):
        A = algebra.randomMV()
        B = algebra.randomMV()
        res = ((A + 1j*B)(2)).value
        res2 = A(2).value + 1j*B(2).value
        np.testing.assert_allclose(res, res2)

    