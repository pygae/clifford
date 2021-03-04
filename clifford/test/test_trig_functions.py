
from clifford import Cl
import numpy as np
import pytest


class TestScalarProperties:
    @pytest.fixture()
    def element(self):
        alg, blades = Cl(0, 0, 0)
        return alg.scalar

    @pytest.mark.parametrize('np_func', [np.sin,
                                         np.cos,
                                         np.tan,
                                         np.sinh,
                                         np.cosh,
                                         np.tanh])
    def test_trig(self, element, np_func):
        for x in np.linspace(-2 * np.pi, 2 * np.pi, 13):
            assert abs(np_func(x*element).value[0] - np_func(x)) < 1E-10


class TestDualNumberProperties:
    @pytest.fixture()
    def element(self):
        alg, blades = Cl(0, 0, 1)
        return alg.scalar, blades['e1']

    @pytest.mark.parametrize('func, deriv_func', [(np.sin, np.cos),
                                                  (np.cos, lambda x: -np.sin(x)),
                                                  (np.tan, lambda x: (1/np.cos(x)**2)),
                                                  (np.sinh, np.cosh),
                                                  (np.cosh, np.sinh),
                                                  (np.tanh, lambda x: (1 - np.tanh(x)**2))])
    def test_derivatives(self, element, func, deriv_func):
        for x in np.linspace(-2 * np.pi, 2 * np.pi, 13):
            result = func(x * element[0] + element[1])
            assert abs(result.value[0] - func(x)) < 1E-10
            assert abs(result.value[1] - deriv_func(x)) < 1E-10


class TestComplexNumberProperties:
    @pytest.fixture()
    def Cl010element(self):
        alg, blades = Cl(0, 1, 0)
        return alg.scalar, blades['e1']

    @pytest.fixture()
    def Cl000element(self):
        alg, blades = Cl(0, 0, 0)
        return alg.MultiVector(0*1j + alg.scalar.value), alg.MultiVector(1j*alg.scalar.value)

    @pytest.mark.parametrize('np_func', [np.sin,
                                         np.cos,
                                         np.tan,
                                         np.sinh,
                                         np.cosh,
                                         np.tanh])
    def test_trig_Cl010(self, Cl010element, np_func):
        """
        This tests the a clifford algebra isomorphic to the complex numbers
        """
        for x in np.linspace(-2 * np.pi, 2 * np.pi, 13):
            for y in np.linspace(-2 * np.pi, 2 * np.pi, 13):
                complex_mv = x * Cl010element[0] + y * Cl010element[1]
                complex_value = x + 1j * y
                result = np_func(complex_mv)
                assert abs(result.value[0] + 1j * result.value[1] - np_func(complex_value)) < 1E-10

    @pytest.mark.parametrize('np_func', [np.sin,
                                         np.cos,
                                         np.tan,
                                         np.sinh,
                                         np.cosh,
                                         np.tanh])
    def test_trig_CxCl000(self, Cl000element, np_func):
        """
        This tests the complexified clifford algebra of only the scalars
        """
        for x in np.linspace(-2 * np.pi, 2 * np.pi, 13):
            for y in np.linspace(-2 * np.pi, 2 * np.pi, 13):
                complex_mv = x * Cl000element[0] + y * Cl000element[1]
                complex_value = x + 1j * y
                result = np_func(complex_mv)
                assert abs(result.value[0] - np_func(complex_value)) < 1E-10
