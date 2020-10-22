import math

import pytest
import numpy.testing as npt

from clifford.cga import CGA


@pytest.fixture(scope='module')
def cga():
    return CGA(3)


@pytest.mark.parametrize('method', [
    CGA.round,
    CGA.flat,
    CGA.translation,
    CGA.transversion,
    CGA.dilation,
    CGA.rotation,
])
def test_random_construction(cga, method):
    method(cga)


@pytest.mark.parametrize('method', [
    CGA.round,
    pytest.param(CGA.flat, marks=pytest.mark.xfail(raises=AttributeError, reason='gh-180')),
    CGA.translation,
    CGA.transversion,
    CGA.dilation,
    CGA.rotation,
])
def test_repr(cga, method):
    obj = method(cga)
    repr(obj)


@pytest.mark.parametrize('method', [
    CGA.round,
    CGA.flat,
    CGA.translation,
    pytest.param(CGA.transversion, marks=pytest.mark.xfail(raises=AssertionError, reason='gh-182')),
    pytest.param(CGA.dilation, marks=pytest.mark.xfail(raises=TypeError, reason='gh-181')),
    CGA.rotation,
])
def test_copy_construction(cga, method):
    # get a random one
    obj = method(cga)

    # and then copy
    obj2 = method(cga, obj.mv)

    assert obj.mv == obj2.mv


@pytest.mark.parametrize('method', [
    CGA.round,
    CGA.flat,
])
def test_dims_construction(cga, method):
    obj = method(cga, 2)
    if hasattr(obj, 'dim'):
        assert obj.dim == 2

    obj = method(cga, 1)
    if hasattr(obj, 'dim'):
        assert obj.dim == 1


@pytest.mark.parametrize('method', [
    CGA.round,
    pytest.param(CGA.flat, marks=pytest.mark.xfail(raises=AssertionError, reason='gh-100'))
])
def test_from_points_construction(cga, method):
    blades = cga.layout.blades
    e1 = blades['e1']
    e2 = blades['e2']
    e3 = blades['e3']
    # can raise / lower without affecting the result
    assert method(cga, e1, e2, e3).mv == method(cga, e1, e2, cga.up(e3)).mv


def test_round_from_center_radius_init(cga):
    blades = cga.layout.blades
    e1 = blades['e1']
    e2 = blades['e2']

    c = cga.round((3*e1+4*e2, 5))
    npt.assert_almost_equal(c.center_down, 3*e1+4*e2)
    npt.assert_almost_equal(c.radius, 5)


def test_round_from_center_radius_method(cga):
    blades = cga.layout.blades
    e1 = blades['e1']
    e2 = blades['e2']

    # TODO: this should be a static method, not a mutator
    c = cga.round()
    c.from_center_radius(30*e1+40*e2, 50)
    npt.assert_almost_equal(c.center_down, 30.*e1+40.*e2)
    npt.assert_almost_equal(c.radius, 50.)


class TestRotation:

    def test_from_bivector(self, cga):
        B = cga.layout.blades['e12']
        cga.rotation(B*math.pi)

    @pytest.mark.parametrize('theta', [
        # produces a pure bivector rotor
        pytest.param(math.pi/2, id='pi/2', marks=[
            pytest.mark.xfail(raises=AssertionError, reason='gh-185')
        ]),
        # produces a pure scalar rotor
        pytest.param(math.pi, id='pi', marks=[
            pytest.mark.xfail(raises=ValueError, reason='gh-185')
        ]),
    ])
    def test_roundtrip(self, cga, theta):
        """ Test that malicious rotations can still be reconstructed from their .mv """
        B = cga.layout.blades['e12']
        r = cga.rotation(B*theta)
        r2 = cga.rotation(r.mv)
        assert r.mv == r2.mv


def test_translation_from_vector(cga):
    blades = cga.layout.blades
    e1 = blades['e1']
    e2 = blades['e2']
    e3 = blades['e3']

    t = cga.translation(e1 + e2)
    assert t(e3) == cga.up(e1 + e2 + e3)
    assert t(cga.up(e3)) == cga.up(e1 + e2 + e3)
