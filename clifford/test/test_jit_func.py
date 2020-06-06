import numpy as np
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

        @jit_func(layout=self.layout, ast_debug=True)
        def test_func(A, B):
            op = (A+B)(1)
            return op

        def slow_test_func(A, B):
            op = (A+B)(1)
            return op

        np.testing.assert_allclose(test_func(e1, e12).value, slow_test_func(e1, e12).value)

    def test_reverse(self):
        e12 = self.blades['e12']

        @jit_func(layout=self.layout, ast_debug=True)
        def test_func(A):
            op = ~A
            return op

        def slow_test_func(A):
            op = ~A
            return op

        np.testing.assert_allclose(test_func(e12).value, slow_test_func(e12).value)

    def test_compound_expression(self):
        e1 = self.blades['e1']
        e2 = self.blades['e2']
        e34 = self.blades['e34']
        einf = self.layout.einf

        @jit_func(layout=self.layout, ast_debug=True)
        def test_func(A, B, C, D):
            op = ~(((A * B) * ~C) | (B ^ A)) - 3.1 - A - 7 * B + 5 + C + 2.5 + (2 ^ (A * B * C) ^ 3) + (A | 5) + (A + D)(2)
            return op

        def slow_test_func(A, B, C, D):
            op = ~(((A * B) * ~C) | (B ^ A)) - 3.1 - A - 7 * B + 5 + C + 2.5 + (2 ^ (A * B * C) ^ 3) + (A | 5) + (A + D)(2)
            return op

        np.testing.assert_allclose(test_func(e1, e2, einf, e34).value, slow_test_func(e1, e2, einf, e34).value)

    def test_benchmark_jit(self, benchmark):
        @jit_func(layout=self.layout)
        def test_func(A, B, C, D):
            op = ~(((A * B) * ~C) | (B ^ A)) - 3.1 - A - 7 * B + 5 + C + 2.5 + (2 ^ (A * B * C) ^ 3) + (A | 5) + (A + D)(2)
            return op
        benchmark(test_func, self.blades['e1'], self.blades['e2'], self.layout.einf, self.blades['e34'])

    def test_benchmark_nojit(self, benchmark):
        def test_func(A, B, C, D):
            op = ~(((A * B) * ~C) | (B ^ A)) - 3.1 - A - 7 * B + 5 + C + 2.5 + (2 ^ (A * B * C) ^ 3) + (A | 5) + (A + D)(2)
            return op
        benchmark(test_func, self.blades['e1'], self.blades['e2'], self.layout.einf, self.blades['e34'])
