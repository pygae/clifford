
from .. import Cl
import numpy as np
import cmath
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
        for x in np.linspace(0, 2*np.pi, 100):
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
        for x in np.linspace(0, 2*np.pi, 10):
            result = func(x * element[0] + element[1])
            assert abs(result.value[0] - func(x)) < 1E-10
            assert abs(result.value[1] - deriv_func(x)) < 1E-10


class TestComplexNumberProperties:
    @pytest.fixture()
    def element(self):
        alg, blades = Cl(0, 1, 0)
        return alg.scalar, blades['e1']

    @pytest.mark.parametrize('np_func,cmath_func', [(np.sin, cmath.sin),
                                                    (np.cos, cmath.cos),
                                                    (np.tan, cmath.tan),
                                                    (np.sinh, cmath.sinh),
                                                    (np.cosh, cmath.cosh),
                                                    (np.tanh, cmath.tanh)])
    def test_trig(self, element, np_func, cmath_func):
        for x in np.linspace(0, 2 * np.pi, 10):
            for y in np.linspace(0, 2 * np.pi, 10):
                complex_mv = x * element[0] + y * element[1]
                complex_value = x + 1j * y
                result = np_func(complex_mv)
                assert abs(result.value[0] + 1j * result.value[1] - cmath_func(complex_value)) < 1E-10
