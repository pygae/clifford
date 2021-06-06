from functools import reduce

import pytest
import numpy as np

from clifford import Layout, BasisVectorIds, general_exp
from clifford import bivector_split, exp


# fixtures make the test a little faster

@pytest.fixture(scope='module')
def sta():
    return Layout([1, 1, 1, -1], ids=BasisVectorIds(['x', 'y', 'z', 't']))

@pytest.fixture(scope='module')
def pga3d():
    return Layout([1, 1, 1, 0], ids=BasisVectorIds(['x', 'y', 'z', 'w']))

@pytest.fixture(scope='module')
def r22():
    return Layout([1, 1, -1, -1])

@pytest.fixture(scope='module')
def r6():
    return Layout([1, 1, 1, 1, 1, 1])


class TestBivectorSplit:
    def test_sta_bivector_split(self, sta):
        ex, ey, ez, et = sta.basis_vectors_lst
        B = 2 * ex * ey + 4 * ez * et  # Rotation plus boost

        Bs, ls = bivector_split(B, roots=True)
        assert Bs == [4 * ez * et, 2 * ex * ey]
        assert np.allclose(ls, [16.0, -4.0])

    def test_pga3d_bivector_split(self, pga3d):
        ex, ey, ez, ew = pga3d.basis_vectors_lst
        B = 2 * ex * ey + 4 * ez * ew  # Rotation plus translation

        Bs, ls = bivector_split(B, roots=True)
        assert Bs == [2 * ex * ey, 4 * ez * ew]
        assert np.allclose(ls, [-4.0, 0.0])

    def test_r6_bivector_split(self, r6):
        e1, e2, e3, e4, e5, e6 = r6.basis_vectors_lst
        B = 2*e1*e2 + 5*e3*e4 + 7*e5*e6

        Bs, ls = bivector_split(B, roots=True)
        assert Bs == [7*e5*e6, 5*e3*e4, 2*e1*e2]
        assert np.allclose(ls, [-49.0, -25.0, -4.0])

    def test_r22_bivector_split(self, r22):
        e1, e2, e3, e4 = r22.basis_vectors_lst
        B = 0.5 * (e1*e2 + e1*e4 - e2*e3 - e3*e4)

        Bs, ls = bivector_split(B, roots=True)
        assert Bs == [0.25 * ((1-1j)*e1*e2 + (1+1j)*e1*e4 + (-1-1j)*e2*e3 + (-1+1j)*e3*e4),
                      0.25 * ((1+1j)*e1*e2 + (1-1j)*e1*e4 + (-1+1j)*e2*e3 + (-1-1j)*e3*e4)]
        assert np.allclose(ls, [0.5j, -0.5j])

class TestExponential:
    def test_sta_exp(self, sta):
        ex, ey, ez, et = sta.basis_vectors_lst
        B = 1 * ex * ey + 2 * ez * et  # Rotation plus boost
        Bs, ls = bivector_split(B, roots=True)

        R = exp(B)
        Rraw = reduce(lambda tot, x: tot*x, [Bi.exp() for Bi in Bs])
        assert np.allclose(Rraw.value, R.value)
        # assert Bs == [4 * ez * et, 2 * ex * ey]
        # assert np.allclose(ls, [16.0, -4.0])

    def test_pga3d_bivector_split(self, pga3d):
        ex, ey, ez, ew = pga3d.basis_vectors_lst
        B = 2 * ex * ey + 4 * ez * ew  # Rotation plus translation
        Bs, ls = bivector_split(B, roots=True)

        R = exp(B)
        Rraw = reduce(lambda tot, x: tot * x, [Bi.exp() for Bi in Bs])
        assert np.allclose(Rraw.value, R.value)

    def test_r6_bivector_split(self, r6):
        e1, e2, e3, e4, e5, e6 = r6.basis_vectors_lst
        B = 2*e1*e2 + 5*e3*e4 + 7*e5*e6
        Bs, ls = bivector_split(B, roots=True)

        R = exp(B)
        Rraw = reduce(lambda tot, x: tot * x, [Bi.exp() for Bi in Bs])
        assert np.allclose(Rraw.value, R.value)

    def test_r22_bivector_split(self, r22):
        e1, e2, e3, e4 = r22.basis_vectors_lst
        B = 0.5 * (e1*e2 + e1*e4 - e2*e3 - e3*e4)
        Bs, ls = bivector_split(B, roots=True)

        R = exp(B)
        Rraw = reduce(lambda tot, x: tot * x, [Bi.exp() for Bi in Bs])
        assert np.allclose(Rraw.value, R.value)