import pickle

import pytest

import clifford
from clifford.tools.classify import (
    classify,
    Direction, Round, Flat, DualFlat,
    Circle, Sphere,
    _GradedTypesMeta
)


def _grade(x):
    gs = x.grades()
    if not gs:
        return 0
    g, = gs
    return g


class _TestBase:
    """ Common tests for both 2d and 3d CGA """
    @pytest.fixture(params=[
        0, 2, 2j
    ])
    def radius(self, request):
        return request.param

    def _test_roundrip(self, cls, g, **kwargs):
        b = cls[g](**kwargs)
        b_rt = classify(b.mv)
        assert isinstance(b_rt, cls)
        assert b.mv == b_rt.mv
        assert _grade(b.mv) == b._grade
        repr(b_rt)  # check this doesn't crash

        # infer the grade argument
        b_infer = cls(**kwargs)
        assert type(b_infer) is type(b)

        # pickle should roundtrip
        b_pickle = pickle.loads(pickle.dumps(b))
        assert type(b_pickle) is type(b)

    def test_roundtrip_direction(self, direction):
        g = _grade(direction)
        self._test_roundrip(
            Direction, g + 1,
            direction=direction,
        )

    def test_roundtrip_round(self, direction, location, radius):
        g = _grade(direction)
        self._test_roundrip(
            Round, g + 1,
            direction=direction, location=location, radius=radius,
        )

    def test_roundtrip_flat(self, direction, location):
        g = _grade(direction)
        self._test_roundrip(
            Flat, g + 2,
            direction=direction, location=location,
        )

    def test_roundtrip_dual_flat(self, direction, location):
        g = _grade(direction)
        self._test_roundrip(
            DualFlat, self.layout.dims - (g + 2),
            flat=Flat(direction=direction, location=location),
        )

    def test_classify_0(self):
        b = classify(0*self.e1)
        assert isinstance(b, Direction[0])

    def test_classify_nonblade(self):
        with pytest.raises(ValueError):
            classify(1 + self.e1)

    def test_classify_base_ga(self):
        with pytest.raises(ValueError):
            classify(self.layout._base_layout.scalar)


class TestCGA2D(_TestBase):
    layout, blades, stuff = clifford.conformalize(clifford.Cl(2)[0])
    e1 = blades['e1']
    e2 = blades['e2']

    @pytest.fixture(params=[
        layout.scalar,
        e1,
        e1^e2,
    ])
    def direction(self, request):
        return request.param

    @pytest.fixture(params=[
        layout.scalar * 0,
        3*e1,
    ])
    def location(self, request):
        return request.param


class TestCGA3D(_TestBase):
    layout, blades, stuff = clifford.conformalize(clifford.Cl(3)[0])
    e1 = blades['e1']
    e2 = blades['e2']
    e3 = blades['e3']

    @pytest.fixture(params=[
        layout.scalar,
        e1,
        e1^e2,
        e1^e2^e3,
    ])
    def direction(self, request):
        return request.param

    @pytest.fixture(params=[
        layout.scalar * 0,
        3*e1,
    ])
    def location(self, request):
        return request.param

    def test_inferred_dims(self):
        """ Test that the appropriate subclass is inferred from arguments """
        e1, e2, e3 = self.e1, self.e2, self.e3
        C = Round(location=e1, direction=e2^e3, radius=1)
        assert isinstance(C, Circle)
        S = Round(location=e1, direction=e1^e2^e3, radius=1)
        assert isinstance(S, Sphere)

        # wrong grade
        with pytest.raises(ValueError):
            Sphere(location=e1, direction=e1^e2, radius=1)
        with pytest.raises(ValueError):
            Circle(location=e1, direction=e1^e2^e3, radius=1)


class TestGradedTypesMeta:

    def test_instantiation(self):
        class GradedThing(metaclass=_GradedTypesMeta):
            pass

        with pytest.raises(TypeError):
            GradedThing()
        with pytest.raises(TypeError):
            GradedThing[1][1]()

        g = GradedThing[1]()
        assert g._grade == 1
        assert isinstance(g, GradedThing[1])
        assert isinstance(g, GradedThing)

        class GradedSubThing(GradedThing):
            pass

        g = GradedSubThing[1]()
        assert isinstance(g, GradedSubThing[1])
        assert isinstance(g, GradedSubThing)
        assert isinstance(g, GradedThing[1])
        assert isinstance(g, GradedThing)

    def test_override(self):
        class GradedThing(metaclass=_GradedTypesMeta):
            pass

        class GradedThingOverride(GradedThing[1], override=True):
            pass

        class GradedThingNotOverride(GradedThing[1]):
            pass

        g = GradedThing[1]()
        assert isinstance(g, GradedThingOverride)
        assert isinstance(g, GradedThing[1])
        assert isinstance(g, GradedThing)

        g = GradedThingNotOverride()
        assert isinstance(g, GradedThingNotOverride)
        assert isinstance(g, GradedThing[1])
        assert isinstance(g, GradedThing)

        with pytest.raises(ValueError):
            class GradedThingOverrideTwo(GradedThing[1], override=True):
                pass
