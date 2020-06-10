import numba
import operator

from clifford.g3c import layout, e1, e2
import clifford as cf
import pytest


@numba.njit
def identity(x):
    return x


class TestBasic:
    """ Test very simple construction and field access """

    def test_roundtrip_layout(self):
        layout_r = identity(layout)
        assert type(layout_r) is type(layout)
        assert layout_r is layout

    def test_roundtrip_mv(self):
        e1_r = identity(e1)
        assert type(e1_r) is type(e1_r)

        # mvs are values, and not preserved by identity
        assert e1_r.layout is e1.layout
        assert e1_r == e1

    def test_piecewise_construction(self):
        @numba.njit
        def negate(a):
            return cf.MultiVector(a.layout, -a.value)

        n_e1 = negate(e1)
        assert n_e1.layout is e1.layout
        assert n_e1 == -e1

        @numba.njit
        def add(a, b):
            return cf.MultiVector(a.layout, a.value + b.value)

        ab = add(e1, e2)
        assert ab == e1 + e2
        assert ab.layout is e1.layout

    def test_constant_multivector(self):
        @numba.njit
        def add_e1(a):
            return cf.MultiVector(a.layout, a.value + e1.value)

        assert add_e1(e2) == e1 + e2

    def test_multivector_shorthand(self):
        @numba.njit
        def double(a):
            return a.layout.MultiVector(a.value*2)

        assert double(e2) == 2 * e2


class TestOperators:
    @pytest.mark.parametrize("op", [
        pytest.param(getattr(operator, op), id=op)
        for op in ['add', 'sub', 'mul', 'xor', 'or_']
    ])
    @pytest.mark.parametrize("a,b", [(e1, e2), (1, e1), (e1, 1),
                                     (0.5, 0.5 * e1), (0.5 * e1, 0.5),
                                     (e1, 0.5), (0.5, e1),
                                     (1, 0.5*e1), (0.5*e1, 1)])
    def test_binary(self, op, a, b):
        @numba.njit
        def overload(a, b):
            return op(a, b)

        ab = op(a, b)
        ab_alt = overload(a, b)
        assert ab == ab_alt
        assert ab.layout is ab_alt.layout
        # numba disagrees with numpy about what type `int` is on windows, so we
        # can't directly compare the dtypes. We only care that the float / int
        # state is kept anyway.
        assert ab.value.dtype.kind == ab_alt.value.dtype.kind

    # `op` is not parametrized, for simplicity we only support MultiVector / scalar.
    @pytest.mark.parametrize("a,b", [(e1, 2), (2.0*e1, 2)])
    def test_truediv(self, a, b):
        op = operator.truediv

        @numba.njit
        def overload(a, b):
            return op(a, b)

        ab = op(a, b)
        ab_alt = overload(a, b)
        assert ab == ab_alt
        assert ab.layout is ab_alt.layout
        # numba disagrees with numpy about what type `int` is on windows, so we
        # can't directly compare the dtypes. We only care that the float / int
        # state is kept anyway.
        assert ab.value.dtype.kind == ab_alt.value.dtype.kind

    @pytest.mark.parametrize("op", [
        pytest.param(getattr(operator, op), id=op)
        for op in ['pos', 'neg', 'invert']
    ])
    @pytest.mark.parametrize("a", [layout.scalar, e1, 0.5*e1, e1^e2, 1 + (e1^e2)])
    def test_unary(self, op, a):
        @numba.njit
        def overload(a):
            return op(a)

        ret = op(a)
        ret_alt = overload(a)
        assert ret == ret_alt
        assert ret.layout is ret_alt.layout
        assert ret.value.dtype == ret_alt.value.dtype
