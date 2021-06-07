from functools import reduce

import pytest
import numpy as np

from clifford import Layout, BasisVectorIds
from clifford.invariant_decomposition import bivector_split, rotor_split, exp, log


# Test some known splits in various algebras.
def sta_split():
    alg = Layout([1, 1, 1, -1], ids=BasisVectorIds(['x', 'y', 'z', 't']))
    ex, ey, ez, et = alg.basis_vectors_lst
    return {'B': 2 * ex * ey + 4 * ez * et,
            'Bs': [4 * ez * et, 2 * ex * ey],
            'ls': [16.0, -4.0],
            'logR': 2 * ex * ey + 4 * ez * et}


def pga3d_split():
    alg = Layout([1, 1, 1, 0], ids=BasisVectorIds(['x', 'y', 'z', 'w']))
    ex, ey, ez, ew = alg.basis_vectors_lst
    return {'B': 2 * ex * ey + 4 * ez * ew,
            'Bs': [2 * ex * ey, 4 * ez * ew],
            'ls': [-4.0, 0.0],
            'logR': 2 * ex * ey + 4 * ez * ew}


def r22_split():
    alg = Layout([1, 1, -1, -1])
    e1, e2, e3, e4 = alg.basis_vectors_lst
    return {'B': 0.5 * (e1*e2 + e1*e4 - e2*e3 - e3*e4),
            'Bs': [0.25 * ((1-1j)*e1*e2 + (1+1j)*e1*e4 + (-1-1j)*e2*e3 + (-1+1j)*e3*e4),
                   0.25 * ((1+1j)*e1*e2 + (1-1j)*e1*e4 + (-1+1j)*e2*e3 + (-1-1j)*e3*e4)],
            'ls': [0.5j, -0.5j],
            'logR': 0.5 * (e1*e2 + e1*e4 - e2*e3 - e3*e4)}


def r6_split():
    alg = Layout([1, 1, 1, 1, 1, 1])
    e1, e2, e3, e4, e5, e6 = alg.basis_vectors_lst
    return {'B': 2*e1*e2 + 5*e3*e4 + 7*e5*e6,
            'Bs': [7*e5*e6, 5*e3*e4, 2*e1*e2],
            'ls': [-49.0, -25.0, -4.0],
            # The log is by no means unique, and this example illustrates it.
            # With the conventions chosen now this is the answer, which is a
            # total of 4 pi away from the input B.
            'logR': 2*e1*e2 + (5 - np.pi)*e3*e4 + (7 - 3*np.pi)*e5*e6}


class TestInvariantDecomposition:
    """ Test the invariant decomposition of bivectors and rotors, and the resulting exp and log functions.
    """
    @pytest.mark.parametrize('known_split', [sta_split(), pga3d_split(), r22_split(), r6_split()])
    def test_known_splits(self, known_split):
        B = known_split['B']
        Bs, ls = bivector_split(B, roots=True)
        # Test the bivector split
        assert Bs == known_split['Bs']
        assert np.allclose(ls, known_split['ls'])

        # Test the exp function
        R = exp(B)
        Rraw = reduce(lambda tot, x: tot * x, [Bi.exp() for Bi in Bs])
        assert np.allclose(Rraw.value, R.value)

        for Bi in Bs:
            # Test if the simple bivectors are exponentiated correctly.
            assert np.allclose(Bi.exp().value, exp(Bi).value)

        # Split R into simple rotors.
        Rs = rotor_split(R)
        # Test simpleness of the Ri
        for Ri in Rs:
            assert Ri == Ri(0) + Ri(2)
        # Test commutativity
        for i, Ri in enumerate(Rs):
            for Rj in Rs[i:]:
                assert Ri*Rj-Rj*Ri == 0

        # Reconstruct R from the rotor_split.
        Rre = reduce(lambda tot, x: tot * x, Rs)
        assert np.allclose(R.value, Rre.value)

        logR = log(R)
        assert logR == known_split['logR']
