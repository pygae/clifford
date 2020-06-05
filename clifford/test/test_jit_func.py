import unittest
import numpy as np
import time
from ..jit_func import jit_func


class TestJITFunc(unittest.TestCase):
    def test_compound_expression(self):
        from clifford.g3c import layout, blades
        e1 = blades['e1']
        e2 = blades['e2']
        einf = layout.einf

        @jit_func(layout=layout, ast_debug=True)
        def test_func(A, B, C):
            op = (((A * B) * C) | (B ^ A)) - 3.1 - A - 7 * B + 5 + C + 2.5 + (2 ^ (A * B * C) ^ 3) + (A | 5)
            return op

        def slow_test_func(A, B, C):
            op = (((A * B) * C) | (B ^ A)) - 3.1 - A - 7 * B + 5 + C + 2.5 + (2 ^ (A * B * C) ^ 3) + (A | 5)
            return op

        np.testing.assert_allclose(test_func(e1, e2, einf).value, slow_test_func(e1, e2, einf).value)

    def test_benchmark(self):
        from clifford.g3c import layout, blades
        e1 = blades['e1']
        e2 = blades['e2']
        einf = layout.einf

        @jit_func(layout=layout)
        def test_func(A, B, C):
            op = (((A * B) * C) | (B ^ A)) - 3.1 - A - 7 * B + 5 + C + 2.5 + (2 ^ (A * B * C) ^ 3) + (A | 5)
            return op

        def slow_test_func(A, B, C):
            op = (((A * B) * C) | (B ^ A)) - 3.1 - A - 7 * B + 5 + C + 2.5 + (2 ^ (A * B * C) ^ 3) + (A | 5)
            return op

        test_func(e1, e2, einf)
        slow_test_func(e1, e2, einf)

        nrepeats = 10000
        start_time = time.time()
        for i in range(nrepeats):
            test_func(e1, e2, einf)
        end_time = time.time()
        print('With jit_func (us): ', 1E6 * (end_time - start_time) / nrepeats)

        nrepeats = 10000
        start_time = time.time()
        for i in range(nrepeats):
            slow_test_func(e1, e2, einf)
        end_time = time.time()
        print('Without jit_func (us): ', 1E6 * (end_time - start_time) / nrepeats)


if __name__ == '__main__':
    unittest.main()
