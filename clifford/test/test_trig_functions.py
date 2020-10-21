
from .. import Cl
import numpy as np
import pytest


class TestScalarProperties:

    @pytest.fixture()
    def element(self):
        alg, blades = Cl(0, 0, 0)
        return blades['']

    def test_sin(self, element):
        for x in np.linspace(0, 2*np.pi, 100):
            assert abs(np.sin(x*element).value[0] - np.sin(x)) < 1E-10

    def test_cos(self, element):
        for x in np.linspace(0, 2*np.pi, 100):
            assert abs(np.cos(x*element).value[0] - np.cos(x)) < 1E-10

    def test_tan(self, element):
        for x in np.linspace(-0.9*np.pi/2, 0.9*np.pi/2, 100):
            assert abs(np.tan(x*element).value[0] - np.tan(x)) < 1E-10

    def test_sinh(self, element):
        for x in np.linspace(0, 2*np.pi, 100):
            assert abs(np.sinh(x*element).value[0] - np.sinh(x)) < 1E-10

    def test_cosh(self, element):
        for x in np.linspace(0, 2*np.pi, 100):
            assert abs(np.cosh(x*element).value[0] - np.cosh(x)) < 1E-10

    def test_tanh(self, element):
        for x in np.linspace(-np.pi, np.pi, 100):
            assert abs(np.tanh(x*element).value[0] - np.tanh(x)) < 1E-10


class TestDualNumberProperties:
    @pytest.fixture()
    def element(self):
        alg, blades = Cl(0, 0, 1)
        return blades[''], blades['e1']

    def test_sin(self, element):
        for x in np.linspace(0, 2*np.pi, 10):
            result = np.sin(x * element[0] + element[1])
            assert abs(result.value[0] - np.sin(x)) < 1E-10
            assert abs(result.value[1] - np.cos(x)) < 1E-10

    def test_cos(self, element):
        for x in np.linspace(0, 2*np.pi, 10):
            result = np.cos(x * element[0] + element[1])
            assert abs(result.value[0] - np.cos(x)) < 1E-10
            assert abs(result.value[1] + np.sin(x)) < 1E-10

    def test_tan(self, element):
        for x in np.linspace(0, 2*np.pi, 10):
            result = np.tan(x * element[0] + element[1])
            assert abs(result.value[0] - np.tan(x)) < 1E-10
            assert abs(result.value[1] - (1/np.cos(x)**2)) < 1E-10

    def test_sinh(self, element):
        for x in np.linspace(0, 2*np.pi, 10):
            result = np.sinh(x * element[0] + element[1])
            assert abs(result.value[0] - np.sinh(x)) < 1E-10
            assert abs(result.value[1] - np.cosh(x)) < 1E-10

    def test_cosh(self, element):
        for x in np.linspace(0, 2*np.pi, 10):
            result = np.cosh(x * element[0] + element[1])
            assert abs(result.value[0] - np.cosh(x)) < 1E-10
            assert abs(result.value[1] - np.sinh(x)) < 1E-10

    def test_tanh(self, element):
        for x in np.linspace(0, 2*np.pi, 10):
            result = np.tanh(x * element[0] + element[1])
            assert abs(result.value[0] - np.tanh(x)) < 1E-10
            assert abs(result.value[1] - (1 - np.tanh(x)**2)) < 1E-10
