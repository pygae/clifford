from functools import reduce, lru_cache
import itertools
import operator

import pytest
import numpy as np

from clifford import Layout, BasisVectorIds
from clifford.invariant_decomposition import bivector_split, rotor_split, exp, log

import clifford as cf
from . import rng  # noqa: F401
from . import too_slow_without_jit


# Test some known splits in various algebras. The target bivector is `B`,
# the expected split is `Bs`, the expected eigenvalues li = Bi**2 are
# given in `ls`. Lastly, the expected logarithm is `logR`.
# We use `lru_cache` here so that these layouts can be reused between tests,
# but are not constructed at all if not needed.
@lru_cache(maxsize=None)
def sta_split():
    alg = Layout([1, 1, 1, -1], ids=BasisVectorIds(['x', 'y', 'z', 't']))
    ex, ey, ez, et = alg.basis_vectors_lst
    return {'B': 2 * ex * ey + 4 * ez * et,
            'Bs': [4 * ez * et, 2 * ex * ey],
            'ls': [16.0, -4.0],
            'logR': 2 * ex * ey + 4 * ez * et}


@lru_cache(maxsize=None)
def pga3d_split():
    alg = Layout([1, 1, 1, 0], ids=BasisVectorIds(['x', 'y', 'z', 'w']))
    ex, ey, ez, ew = alg.basis_vectors_lst
    return {'B': 2 * ex * ey + 4 * ez * ew,
            'Bs': [4 * ez * ew, 2 * ex * ey],
            'ls': [0.0, -4.0],
            'logR': 2 * ex * ey + 4 * ez * ew}


@lru_cache(maxsize=None)
def r22_split():
    alg = Layout([1, 1, -1, -1])
    e1, e2, e3, e4 = alg.basis_vectors_lst
    return {'B': 0.5 * (e1*e2 + e1*e4 - e2*e3 - e3*e4),
            'Bs': [0.25 * ((1-1j)*e1*e2 + (1+1j)*e1*e4 + (-1-1j)*e2*e3 + (-1+1j)*e3*e4),
                   0.25 * ((1+1j)*e1*e2 + (1-1j)*e1*e4 + (-1+1j)*e2*e3 + (-1-1j)*e3*e4)],
            'ls': [0.5j, -0.5j],
            'logR': 0.5 * (e1*e2 + e1*e4 - e2*e3 - e3*e4)}


@lru_cache(maxsize=None)
def r6_split():
    alg = Layout([1, 1, 1, 1, 1, 1])
    e1, e2, e3, e4, e5, e6 = alg.basis_vectors_lst
    return {'B': 2*e1*e2 + 5*e3*e4 + 7*e5*e6,
            'Bs': [2*e1*e2, 5*e3*e4, 7*e5*e6],
            'ls': [-4.0, -25.0, -49.0],
            # The log is by no means unique, and this example illustrates it.
            # With the conventions of this implementation this is the answer, which is a
            # total of 4 pi away from the input B.
            'logR': (2 - np.pi)*e1*e2 + (5 - np.pi)*e3*e4 + (7 - 2*np.pi)*e5*e6}


@lru_cache(maxsize=None)
def r4_split():
    alg = Layout([1, 1, 1, 1])
    e1, e2, e3, e4 = alg.basis_vectors_lst
    delta = 1
    return {'B': 2*e1*e2 + (2+delta)*e3*e4,
            'Bs': [(2+delta)*e3*e4, 2*e1*e2],
            'ls': [-(2+delta)**2, -4.0],
            'logR': (2-np.pi)*e1*e2 + (2+delta-np.pi)*e3*e4}


class TestInvariantDecomposition:
    """ Test the invariant decomposition of bivectors and rotors, and the resulting exp and log functions.
    """
    @pytest.mark.parametrize('known_split', [
        pytest.param(sta_split, id='sta'),
        pytest.param(pga3d_split, id='pga'),
        pytest.param(r22_split, id='r22'),
        pytest.param(r6_split, id='r6'),
        pytest.param(r4_split, id='r4', marks=[
            pytest.mark.xfail(reason='unknown')]),
    ])
    def test_known_splits(self, known_split):
        known_split = known_split()

        B = known_split['B']
        Bs, ls = bivector_split(B, roots=True)
        # Test the bivector split
        for calculated, known in zip(Bs, known_split['Bs']):
            np.testing.assert_allclose(calculated.value, known.value, atol=1E-5, rtol=1E-5)
        np.testing.assert_allclose(ls, known_split['ls'])

        # Test the exp function agrees with the taylor expansion used by `.exp()`
        R = exp(B)
        Rraw = reduce(operator.mul, [Bi.exp() for Bi in Bs])
        np.testing.assert_allclose(Rraw.value, R.value, atol=1E-5, rtol=1E-5)

        for Bi in Bs:
            # Test if the simple bivectors are exponentiated correctly.
            np.testing.assert_allclose(Bi.exp().value, exp(Bi).value, atol=1E-5, rtol=1E-5)

    @pytest.mark.parametrize('known_split', [
        pytest.param(sta_split, id='sta'),
        pytest.param(pga3d_split, id='pga'),
        pytest.param(r22_split, id='r22'),
        pytest.param(r6_split, id='r6'),
        pytest.param(r4_split, id='r4'),
    ])
    def test_known_rotor_splits(self, known_split):
        known_split = known_split()
        B = known_split['B']

        R = exp(B)

        # Split R into simple rotors.
        Rs = rotor_split(R)
        # Test simpleness of the Ri
        for Ri in Rs:
            np.testing.assert_allclose(Ri.value, Ri(0, 2).value, atol=1E-5, rtol=1E-5)
        # Test commutativity
        for Ri, Rj in itertools.combinations(Rs, 2):
            np.testing.assert_allclose((Ri*Rj).value, (Rj*Ri).value, atol=1E-5, rtol=1E-5)

        # Reconstruct R from the rotor_split.
        Rre = reduce(operator.mul, Rs, B.layout.scalar)
        np.testing.assert_allclose(R.value, Rre.value, atol=1E-5, rtol=1E-5)

        logR = log(R)
        np.testing.assert_allclose(logR.value, known_split['logR'].value, atol=1E-5, rtol=1E-5)

    @pytest.mark.parametrize('r', range(2))
    @pytest.mark.parametrize('p, q', [
        pytest.param(p, total_dims - p,
                     marks=[pytest.mark.veryslow, too_slow_without_jit] if total_dims >= 6 else [])
        for total_dims in [0, 1, 2, 3, 4, 5, 6, 7, 8]
        for p in range(total_dims + 1)
    ])
    def test_unknown_splits(self, p, q, r, rng):  # noqa: F811
        Ntests = 10
        layout, blades = cf.Cl(p, q, r)
        for i in range(Ntests):
            B = layout.randomMV(rng=rng, grades=[2])
            Bs, ls = bivector_split(B, roots=True)
            for Bi, li in zip(Bs, ls):
                # To be simple, you must square to a scalar.
                Bisq = Bi**2
                np.testing.assert_allclose(Bisq.value, Bisq(0).value,
                                           rtol=1E-5, atol=1E-5)
                np.testing.assert_allclose(Bisq.value[0], li,
                                           rtol=1E-5, atol=1E-5)

            # Assert that the bivectors sum to the original
            np.testing.assert_allclose(sum(Bs).value, B.value,
                                       rtol=1E-5, atol=1E-5)

            # Assert that the bivectors are commutative
            for x, y in itertools.combinations(Bs, 2):
                np.testing.assert_allclose((x*y).value, (y*x).value,
                                           rtol=1E-5, atol=1E-5)

            R = exp(B)

            # Make the absolute tolerance of the rotor tests dependent on total dimension
            default_atol = 1E-5
            if p + q + r > 7:
                default_atol = 1E-2

            # Assert that the rotor split multiplies into the original.
            Rs, ls = rotor_split(R, roots=True)
            Rre = reduce(operator.mul, Rs, 1)
            np.testing.assert_allclose(R.value, Rre.value,
                                       rtol=1E-5, atol=default_atol)
            # Assert that the exp(log(R)) is the original rotor R.
            logR = log(R)
            np.testing.assert_allclose(R.value, exp(logR).value,
                                       rtol=1E-5, atol=default_atol)
