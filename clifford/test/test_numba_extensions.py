import operator
import pickle

import numba
import numpy as np
import pytest

from clifford.g3c import layout, e1, e2, e3, e4, e5
import clifford as cf


@numba.njit
def identity(x):
    return x


def assert_mv_equal(a, b):
    assert a.layout is b.layout
    # numba disagrees with numpy about what type `int` is on windows, so we
    # can't directly compare the dtypes. We only care that the float / int
    # state is kept anyway.
    assert a.value.dtype.kind == b.value.dtype.kind
    assert a == b


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
        assert_mv_equal(e1_r, e1)

    def test_piecewise_construction(self):
        @numba.njit
        def negate(a):
            return cf.MultiVector(a.layout, -a.value)

        n_e1 = negate(e1)
        assert n_e1.layout is e1.layout
        assert_mv_equal(n_e1, -e1)

        @numba.njit
        def add(a, b):
            return cf.MultiVector(a.layout, a.value + b.value)

        assert_mv_equal(add(e1, e2), e1 + e2)

    def test_constant_multivector(self):
        @numba.njit
        def add_e1(a):
            return cf.MultiVector(a.layout, a.value + e1.value)

        assert_mv_equal(add_e1(e2), e1 + e2)


class TestLayout:

    def test_multivector_shorthand(self):
        @numba.njit
        def double(a):
            return a.layout.MultiVector(a.value*2)

        @numba.njit
        def czeros(l):
            return l.MultiVector(dtype=np.complex64)

        assert_mv_equal(double(e2), 2 * e2)
        assert_mv_equal(czeros(layout), layout.MultiVector(dtype=np.complex64))

    def test_sig(self):
        @numba.njit
        def jit_func(a):
            return layout.sig

        assert list(jit_func(layout)) == list(layout.sig)

    def test_dims(self):
        @numba.njit
        def jit_func(a):
            return layout.dims

        assert jit_func(layout) == layout.dims

    def test_gaDims(self):
        @numba.njit
        def jit_func(a):
            return layout.gaDims

        assert jit_func(layout) == layout.gaDims


class TestMultiVectorOperators:
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

        assert_mv_equal(op(a, b), overload(a, b))

    # `op` is not parametrized, for simplicity we only support MultiVector / scalar.
    @pytest.mark.parametrize("a,b", [(e1, 2), (2.0*e1, 2)])
    def test_truediv(self, a, b):
        op = operator.truediv

        @numba.njit
        def overload(a, b):
            return op(a, b)

        assert_mv_equal(op(a, b), overload(a, b))

    @pytest.mark.parametrize("op", [
        pytest.param(getattr(operator, op), id=op)
        for op in ['pos', 'neg', 'invert']
    ])
    @pytest.mark.parametrize("a", [layout.scalar, e1, 0.5*e1, e1^e2, 1 + (e1^e2)])
    def test_unary(self, op, a):
        @numba.njit
        def overload(a):
            return op(a)

        assert_mv_equal(op(a), overload(a))

    # We have a special overload for literal arguments for speed
    def literal_grade_func(a, grade):
        @numba.njit
        def grade_func(a):
            return a(grade)
        return grade_func(a)

    @numba.njit
    def grade_func(a, grade):
        return a(grade)

    # vararg versions
    def literal_grade_func_star(a, *grades):
        @numba.njit
        def grade_func(a):
            return a(*grades)
        return grade_func(a)

    @numba.njit
    def grade_func_star(a, *grades):
        return a(*grades)

    @pytest.mark.parametrize('func', [
        literal_grade_func, grade_func,
        literal_grade_func_star, grade_func_star,
    ])
    def test_grade_projection(self, func):
        a = 1 + e1 + (e1^e2) + (e1^e2^e3) + (e1^e2^e3^e4) + (e1^e2^e3^e4^e5)

        assert func(a, 0) == 1
        assert func(a, 1) == e1
        assert func(a, 2) == e1^e2
        assert func(a, 3) == e1^e2^e3
        assert func(a, 4) == e1^e2^e3^e4
        assert func(a, 5) == e1^e2^e3^e4^e5

    @pytest.mark.parametrize('func', [
        literal_grade_func_star, grade_func_star,
    ])
    def test_multiple_grade_projection(self, func):
        a = 1 + e1 + (e1^e2) + (e1^e2^e3) + (e1^e2^e3^e4) + (e1^e2^e3^e4^e5)

        assert func(a, 0, 1) == 1 + e1
        assert func(a, 0, 1, 2, 3, 4, 5) == a


class TestMultiVectorMethods:
    def test_leftLaInv(self):
        @numba.njit
        def jit_func(a):
            return a.leftLaInv()

        assert_mv_equal(jit_func(2.0*e1), (2.0*e1).leftLaInv())

        with pytest.raises(ValueError):
            jit_func(1.0*e1 - e5)

    def test_shirokov_inverse(self):
        @numba.njit
        def jit_func(a):
            return a.shirokov_inverse()

        assert_mv_equal(jit_func(2.0*e1), (2.0*e1).shirokov_inverse())

        with pytest.raises(ValueError):
            jit_func(1.0*e1 - e5)

    def test_hitzer_inverse(self):
        @numba.njit
        def jit_func(a):
            return a.hitzer_inverse()

        assert_mv_equal(jit_func(2.0*e1), (2.0*e1).hitzer_inverse())

        with pytest.raises(ValueError):
            jit_func(1.0*e1 - e5)

    @pytest.mark.parametrize('a', [1 + e1 + e2*e3, 1.0 + e1])
    def test_gradeInvol(self, a):
        @numba.njit
        def jit_func(a):
            return a.gradeInvol()

        assert_mv_equal(a.gradeInvol(), jit_func(a))

    @pytest.mark.parametrize('a,b', [(e1*e2, e2*e3), (e1*e2*1.0, e2*e3)])
    def test_commutator(self, a, b):
        @numba.njit
        def jit_func(a, b):
            return a.commutator(b)

        assert_mv_equal(a.commutator(b), jit_func(a, b))

    @pytest.mark.parametrize('a,b', [(e1*e2, e2*e3), (e1*e2*1.0, e2*e3)])
    def test_anticommutator(self, a, b):
        @numba.njit
        def jit_func(a, b):
            return a.anticommutator(b)

        assert_mv_equal(a.anticommutator(b), jit_func(a, b))


class TestMultiVectorProperties:

    @pytest.mark.parametrize('v', [1 + e1 + e2*e3, 1.0 + e1])
    def test_even(self, v):
        @numba.njit
        def jit_func(a):
            return a.even
        assert_mv_equal(v.even, jit_func(v))

    @pytest.mark.parametrize('v', [1 + e1 + e2*e3, 1.0 + e1])
    def test_odd(self, v):
        @numba.njit
        def jit_func(a):
            return a.odd
        assert_mv_equal(v.odd, jit_func(v))


def test_pickling():
    lt = layout._numba_type_
    assert pickle.loads(pickle.dumps(lt)) is lt

    # gh-349, populating `lt._cache` should not break pickling
    e1._numba_type_  # int
    (e1 * 1.0)._numba_type_  # float
    assert pickle.loads(pickle.dumps(lt)) is lt


def test_A_order():
    import numpy as np

    @numba.njit
    def mul_mv(mv):
        return mv*e3

    # A-order
    mva = layout.MultiVector(np.ones(layout.gaDims))
    mva.value = mva.value[::-1]
    assert not mva.value.flags.c_contiguous
    res_mva = mul_mv(mva)

    # C-order
    mvc = layout.MultiVector(np.ones(layout.gaDims))
    assert mvc.value.flags.c_contiguous
    res_mvc = mul_mv(mvc)

    assert res_mva == res_mvc
