import textwrap

import numpy as np
import pytest
from IPython.lib import pretty

from clifford import Layout, ConformalLayout, BasisVectorIds
from clifford import transformations


# fixtures make the test a little faster

@pytest.fixture(scope='module')
def g2():
    return Layout([1, 1], ids=BasisVectorIds(['u', 'v']))


@pytest.fixture(scope='module')
def g2c(g2):
    return ConformalLayout([1, 1, 1, -1], ids=BasisVectorIds(['u', 'v', 'p', 'm']), layout=g2)


@pytest.fixture(scope='module')
def g3():
    return Layout([1, 1, 1], ids=BasisVectorIds(['x', 'y', 'z']))


class TestBetweenBasisVectors:
    def test_same_layout(self, g3):
        ex, ey, ez = g3.basis_vectors_lst
        x = ex + 2*ey + 4*ex*ez

        f = transformations.between_basis_vectors(g3, g3)
        assert f(x) == x

        f_swap = transformations.between_basis_vectors(g3, g3, {'x': 'y', 'y': 'z', 'z': 'x'})
        assert f_swap(x) == ey + 2*ez + 4*ey*ex

    def test_different_layout(self, g2, g2c, g3):
        e1, e2 = g2.basis_vectors_lst
        e1c, e2c, epc, emc = g2c.basis_vectors_lst

        f = transformations.between_basis_vectors(g2, g2c)
        x = 1 + 2*e1 + 3*e2 + 4*e1*e2
        xc = 1 + 2*e1c + 3*e2c + 4*e1c*e2c
        assert f(x) == xc
        assert f.adjoint(xc) == x

        ex, ey, ez = g3.basis_vectors_lst
        f = transformations.between_basis_vectors(g2, g3, {'u': 'z', 'v': 'y'})
        xs = 1 + 2*ez + 3*ey + 4*ez*ey
        assert f(x) == xs
        assert f.adjoint(xs) == x

        # incorrect layout
        with pytest.raises(ValueError):
            f(xs)

        # invalid basis
        with pytest.raises(ValueError):
            transformations.between_basis_vectors(g2, g3, {'fake': 'z', 'v': 'y'})
        with pytest.raises(ValueError):
            transformations.between_basis_vectors(g2, g3, {'u': 'z', 'v': 'fake'})


class TestOutermorphismMatrix:
    def test_invariants(self, g2):
        e1, e2 = g2.basis_vectors_lst

        matrix = np.array([[ 0, 1],
                           [-1, 0]])
        f = transformations.OutermorphismMatrix(matrix, g2)

        # test the vector transform is as requested
        assert f(e1) == -e2
        assert f(e2) == e1
        # test the generalization is correct
        assert f(e1^e2) == f(e1)^f(e2)
        assert f(g2.scalar) == g2.scalar

        # test that distributivity is respected
        assert f(g2.scalar + 2*e1 + 3*(e1^e2)) == f(g2.scalar) + 2*f(e1) + 3*f(e1^e2)

        assert pretty.pretty(f) == textwrap.dedent("""\
        OutermorphismMatrix(array([[ 0,  1],
                                   [-1,  0]]),
                            Layout([1, 1],
                                   ids=BasisVectorIds(['u', 'v']),
                                   order=BasisBladeOrder.shortlex(2),
                                   names=['', 'eu', 'ev', 'euv']))""")

    def test_between_layouts(self, g2, g3):
        matrix = np.array([[1, 0],
                           [0, 1],
                           [0, 0]])
        # arguments backwards
        with pytest.raises(ValueError):
            transformations.OutermorphismMatrix(matrix, g3, g2)

        e1, e2 = g2.basis_vectors_lst
        ex, ey, ez = g3.basis_vectors_lst

        f = transformations.OutermorphismMatrix(matrix, g2, g3)
        assert f(e1) == ex
        assert f(e2) == ey

        assert pretty.pretty(f) == textwrap.dedent("""\
        OutermorphismMatrix(array([[1, 0],
                                   [0, 1],
                                   [0, 0]]),
                            layout_src=Layout([1, 1],
                                              ids=BasisVectorIds(['u', 'v']),
                                              order=BasisBladeOrder.shortlex(2),
                                              names=['', 'eu', 'ev', 'euv']),
                            layout_dst=Layout([1, 1, 1],
                                              ids=BasisVectorIds(['x', 'y', 'z']),
                                              order=BasisBladeOrder.shortlex(3),
                                              names=['', 'ex', 'ey', 'ez', 'exy', 'exz', 'eyz', 'exyz']))""")


class TestLinearMatrix:
    def test_same_layout(self, g2):
        e1, e2 = g2.basis_vectors_lst

        dual_matrix = np.array([
            [ 0,  0, 0, 1],
            [ 0,  0, 1, 0],
            [ 0, -1, 0, 0],
            [-1,  0, 0, 0]
        ])
        f_dual = transformations.LinearMatrix(dual_matrix, g2)
        x = 1 + 2*e2 + 3*e1*e2
        assert f_dual(x) == x.dual()

        assert pretty.pretty(f_dual) == textwrap.dedent("""\
        LinearMatrix(array([[ 0,  0,  0,  1],
                            [ 0,  0,  1,  0],
                            [ 0, -1,  0,  0],
                            [-1,  0,  0,  0]]),
                     Layout([1, 1],
                            ids=BasisVectorIds(['u', 'v']),
                            order=BasisBladeOrder.shortlex(2),
                            names=['', 'eu', 'ev', 'euv']))""")

    def test_between_layouts(self, g2, g3):
        matrix = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 1]])

        # arguments backwards
        with pytest.raises(ValueError):
            transformations.LinearMatrix(matrix, g3, g2)

        e1, e2 = g2.basis_vectors_lst
        ex, ey, ez = g3.basis_vectors_lst

        f = transformations.LinearMatrix(matrix, g2, g3)
        assert f(e1) == ex
        assert f(e2) == ey
        assert f(e1^e2) == ex^ey^ez

    def test_from_function(self, g2, g3):
        # passing tests are within the doctests

        # wrong implicit layout
        with pytest.raises(ValueError):
            def bad_f(e):
                # these paths return different layouts!
                if e.grades() == {0}:
                    return g3.scalar
                else:
                    return e
            transformations.LinearMatrix.from_function(bad_f, g2)

        # wrong explicit layout
        with pytest.raises(ValueError):
            transformations.LinearMatrix.from_function(lambda x: x, g2, g3)
