import numpy as np
import pytest
from ..jit_func import jit_func


class TestJITFunc:

    @classmethod
    def setup_class(cls):
        from clifford.g3c import layout, blades
        cls.layout = layout
        cls.blades = blades

    def test_grade_selection(self):
        e1 = self.blades['e1']
        e12 = self.blades['e12']

        def slow_test_func(A, B):
            op = (A(1)+B(2))(1)
            return op

        test_func = jit_func(self.layout, ast_debug=True)(slow_test_func)

        np.testing.assert_allclose(test_func(e1, e12).value, slow_test_func(e1, e12).value)

    def test_reverse(self):
        e12 = self.blades['e12']

        def slow_test_func(A):
            op = ~A
            return op

        test_func = jit_func(self.layout, ast_debug=True)(slow_test_func)

        np.testing.assert_allclose(test_func(e12).value, slow_test_func(e12).value)

    def test_compound_expression(self):
        e1 = self.blades['e1']
        e2 = self.blades['e2']
        e34 = self.blades['e34']
        e12345 = self.blades['e12345']
        einf = self.layout.einf

        pi = np.pi
        def slow_test_func(A, B, C, D):
            op = ~(((A * B) * ~C) | (B ^ A)) - 3.1 - A - 7 * B + 5 + C + 2.5 + (2 ^ (A * B * C) ^ 3) + (A | 5) + (A + D)(2) + pi*e12345
            return op

        test_func = jit_func(self.layout,
                             mv_constants={'e12345': e12345},
                             scalar_constants={'pi': pi}
                             )(slow_test_func)

        np.testing.assert_allclose(test_func(e1, e2, einf, e34).value, slow_test_func(e1, e2, einf, e34).value)

    @pytest.mark.parametrize('use_jit', [False, True])
    def test_benchmark(self, use_jit, benchmark):
        e12345 = self.blades['e12345']
        pi = np.pi
        def test_func(A, B, C, D):
            op = ~(((A * B) * ~C) | (B ^ A)) - 3.1 - A - 7 * B + 5 + C + 2.5 + (2 ^ (A * B * C) ^ 3) + (A | 5) + (A + D)(2) + pi*e12345
            return op

        if use_jit:
            test_func = jit_func(self.layout,
                                 mv_constants={'e12345': e12345},
                                 scalar_constants={'pi': pi}
                                 )(test_func)
        benchmark(test_func, self.blades['e1'], self.blades['e2'], self.layout.einf, self.blades['e34'])

