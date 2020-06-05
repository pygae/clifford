import numpy as np
import time
from ..jit_func import jit_func


class TestJITFunc:

    def test_grade_selection(self):
        from clifford.g3c import layout, blades
        e1 = blades['e1']
        e12 = blades['e12']

        @jit_func(layout=layout, ast_debug=True)
        def test_func(A, B):
            op = (A+B)(1)
            return op

        def slow_test_func(A, B):
            op = (A+B)(1)
            return op

        np.testing.assert_allclose(test_func(e1, e12).value, slow_test_func(e1, e12).value)


    def test_reverse(self):
        from clifford.g3c import layout, blades
        e12 = blades['e12']

        @jit_func(layout=layout, ast_debug=True)
        def test_func(A):
            op = ~A
            return op

        def slow_test_func(A):
            op = ~A
            return op

        np.testing.assert_allclose(test_func(e12).value, slow_test_func(e12).value)

    def test_compound_expression(self):
        from clifford.g3c import layout, blades
        e1 = blades['e1']
        e2 = blades['e2']
        e34 = blades['e34']
        einf = layout.einf

        @jit_func(layout=layout, ast_debug=True)
        def test_func(A, B, C, D):
            op = ~(((A * B) * ~C) | (B ^ A)) - 3.1 - A - 7 * B + 5 + C + 2.5 + (2 ^ (A * B * C) ^ 3) + (A | 5) + (A + D)(2)
            return op

        def slow_test_func(A, B, C, D):
            op = ~(((A * B) * ~C) | (B ^ A)) - 3.1 - A - 7 * B + 5 + C + 2.5 + (2 ^ (A * B * C) ^ 3) + (A | 5) + (A + D)(2)
            return op

        np.testing.assert_allclose(test_func(e1, e2, einf, e34).value, slow_test_func(e1, e2, einf, e34).value)

    def test_benchmark(self):
        from clifford.g3c import layout, blades
        e1 = blades['e1']
        e2 = blades['e2']
        e34 = blades['e34']
        einf = layout.einf

        @jit_func(layout=layout)
        def test_func(A, B, C, D):
            op = ~(((A * B) * ~C) | (B ^ A)) - 3.1 - A - 7 * B + 5 + C + 2.5 + (2 ^ (A * B * C) ^ 3) + (A | 5) + (A + D)(2)
            return op

        def slow_test_func(A, B, C, D):
            op = ~(((A * B) * ~C) | (B ^ A)) - 3.1 - A - 7 * B + 5 + C + 2.5 + (2 ^ (A * B * C) ^ 3) + (A | 5) + (A + D)(2)
            return op

        print(test_func(e1, e2, einf, e34))
        print(slow_test_func(e1, e2, einf, e34))

        nrepeats = 10000
        start_time = time.time()
        for i in range(nrepeats):
            test_func(e1, e2, einf, e34)
        end_time = time.time()
        print('With jit_func (us): ', 1E6 * (end_time - start_time) / nrepeats)

        nrepeats = 10000
        start_time = time.time()
        for i in range(nrepeats):
            slow_test_func(e1, e2, einf, e34)
        end_time = time.time()
        print('Without jit_func (us): ', 1E6 * (end_time - start_time) / nrepeats)

