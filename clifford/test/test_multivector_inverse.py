import numpy as np
import pytest

import clifford as cf
from . import rng  # noqa: F401

from clifford._numba_utils import DISABLE_JIT


too_slow_without_jit = pytest.mark.skipif(
    DISABLE_JIT, reason="test is too slow without JIT"
)


class TestClosedForm:

    @pytest.mark.parametrize('p, q, r', [
        (p, p_add_q - p, r)
        for p_add_q in [1, 2, 3, 4, 5]
        for p in range(p_add_q + 1)
        for r in range(2)
        if r + p_add_q <= 5  # beyond this the hitzer_inverse is not supported
    ])
    def test_hitzer_inverse(self, p, q, r, rng):  # noqa: F811
        Ntests = 10
        layout, blades = cf.Cl(p, q, r)
        for i in range(Ntests):
            mv = layout.randomMV(rng=rng)
            mv_inv = mv.hitzer_inverse()
            np.testing.assert_allclose((mv * mv_inv).value,
                                       layout.scalar.value,
                                       rtol=1E-5,
                                       atol=1E-6)

    @pytest.mark.parametrize('r', range(2))
    @pytest.mark.parametrize('p, q', [
        pytest.param(p, total_dims - p, marks=[pytest.mark.slow, too_slow_without_jit] if total_dims >= 6 else [])
        for total_dims in [1, 2, 3, 4, 5, 6, 7, 8]
        for p in range(total_dims + 1)
    ])
    def test_shirokov_inverse(self, p, q, r, rng):  # noqa: F811
        Ntests = 5
        layout, blades = cf.Cl(p, q, r)
        if p + q + r > 7:
            # This is perhaps a little extreme but algorithm really does struggle
            # with larger algebras
            atol = 0.15
        else:
            atol = 1E-6
        for i in range(Ntests):
            mv = layout.randomMV(rng=rng)
            mv_inv = mv.shirokov_inverse()
            np.testing.assert_allclose((mv * mv_inv).value,
                                       layout.scalar.value,
                                       rtol=1E-5,
                                       atol=atol)
