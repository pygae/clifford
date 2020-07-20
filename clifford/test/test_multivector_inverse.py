import numpy as np
import pytest

import clifford as cf


class TestClosedForm:

    @pytest.mark.parametrize('p, q', [
        (p, total_dims - p)
        for total_dims in [1, 2, 3, 4, 5]
        for p in range(total_dims + 1)
    ])
    def test_hitzer_inverse(self, p, q):
        Ntests = 100
        layout, blades = cf.Cl(p, q)
        for i in range(Ntests):
            mv = layout.randomMV()
            mv_inv = mv.hitzer_inverse()
            assert np.all(np.abs(((mv * mv_inv) - 1.).value) < 1.e-11)
