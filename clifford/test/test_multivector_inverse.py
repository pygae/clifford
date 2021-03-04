import numpy as np
import pytest

import clifford as cf
from . import rng  # noqa: F401


class TestClosedForm:

    @pytest.mark.parametrize('p, q', [
        (p, total_dims - p)
        for total_dims in [1, 2, 3, 4, 5]
        for p in range(total_dims + 1)
    ])
    def test_hitzer_inverse(self, p, q, rng):  # noqa: F811
        Ntests = 100
        layout, blades = cf.Cl(p, q)
        for i in range(Ntests):
            mv = layout.randomMV(rng=rng)
            mv_inv = mv.hitzer_inverse()
            np.testing.assert_almost_equal((mv * mv_inv).value,
                                           layout.scalar.value)
