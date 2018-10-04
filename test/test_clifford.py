from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
from past.builtins import range

from clifford import Cl, randomMV, Frame, get_mult_function, conformalize, grade_obj


import numpy as np
from numpy import exp, float64, testing
import unittest
from functools import reduce

from nose.plugins.skip import SkipTest


class CliffordTests(unittest.TestCase):

    def setUp(self):
        self.algebras = [Cl(i) for i in [3, 4, 5]]

    @SkipTest
    def test_inverse(self):
        for layout, blades in self.algebras:
            a = 1. + blades['e1']
            self.assertRaises(ValueError, lambda x: 1/x, a)
            for i in range(10):
                a = randomMV(layout, grades=[0, 1])
                denominator = float(a(1)**2-a(0)**2)
                if abs(denominator) > 1.e-5:
                    a_inv = (-a(0)/denominator) + ((1./denominator) * a(1))
                    self.assert_(abs((a * a_inv)-1.) < 1.e-11)
                    self.assert_(abs((a_inv * a)-1.) < 1.e-11)
                    self.assert_(abs(a_inv - 1./a) < 1.e-11)

    def test_exp(self):

        layout, blades = self.algebras[0]
        R = exp(blades['e12'])
        e1 = blades['e1']
        R*e1*~R

    def test_indexing(self):
        layout, blades = self.algebras[0]
        e12 = blades['e12']
        e1 = blades['e1']
        e2 = blades['e2']
        e3 = blades['e3']
        self.assertAlmostEqual(e12[e12],1)
        self.assertAlmostEqual(e12[e3], 0)
        self.assertAlmostEqual(e12[(2,1)], -1)


    def test_add_float64(self):
        '''
        test array_wrap method to take control addition from numpy array
        '''
        layout, blades = self.algebras[0]
        e1 = blades['e1']

        float64(1) + e1
        self.assertEqual(1 + e1, float64(1) + e1)


class BasicConformal41Tests(unittest.TestCase):
    def test_metric(self):
        layout = Cl(4, 1)[0]
        e1 = layout.blades['e1']
        e2 = layout.blades['e2']
        e3 = layout.blades['e3']
        e4 = layout.blades['e4']
        e5 = layout.blades['e5']

        self.assertAlmostEqual((e1 * e1)[0], 1)
        self.assertAlmostEqual((e2 * e2)[0], 1)
        self.assertAlmostEqual((e3 * e3)[0], 1)
        self.assertAlmostEqual((e4 * e4)[0], 1)
        self.assertAlmostEqual((e5 * e5)[0], -1)


    def test_factorise(self):
        layout_a = Cl(3)[0]
        layout,blades,stuff = conformalize(layout_a)
        e1 = layout.blades['e1']
        e2 = layout.blades['e2']
        e3 = layout.blades['e3']
        e4 = layout.blades['e4']
        e5 = layout.blades['e5']

        up = stuff['up']

        blade = up(e1 + 3*e2 + 4*e3)^up(5*e1 + 3.3*e2 + 10*e3)^up(-13.1*e1)

        basis, scale = blade.factorise()
        new_blade = (reduce(lambda a, b: a^b, basis)*scale)
        print(new_blade)
        print(blade)
        np.testing.assert_almost_equal(new_blade.value, blade.value, 5)


    def test_gp_op_ip(self):
        layout = Cl(4, 1)[0]
        e1 = layout.blades['e1']
        e2 = layout.blades['e2']
        e3 = layout.blades['e3']
        e4 = layout.blades['e4']
        e5 = layout.blades['e5']

        e123 = layout.blades['e123']
        np.testing.assert_almost_equal(e123.value, (e1 ^ e2 ^ e3).value)
        np.testing.assert_almost_equal(e123.value, (e1 * e2 * e3).value)

        e12345 = layout.blades['e12345']
        np.testing.assert_almost_equal(e12345.value, (e1 ^ e2 ^ e3 ^ e4 ^ e5).value)
        np.testing.assert_almost_equal(e12345.value, (e1 * e2 * e3 * e4 * e5).value)

        e12 = layout.blades['e12']
        np.testing.assert_almost_equal(-e12.value, (e2 ^ e1).value)

        t = np.zeros(32)
        t[0] = -1
        np.testing.assert_almost_equal(t, (e12*e12).value)


class BasicSpaceTimeTests(unittest.TestCase):
    def test_initialise(self):

        # Dirac Algebra  `D`
        D, D_blades = Cl(1, 3, names='d', firstIdx=0)

        # Pauli Algebra  `P`
        P, P_blades = Cl(3, names='p')

        # put elements of each in namespace
        locals().update(D_blades)
        locals().update(P_blades)


class BasicAlgebraTests(unittest.TestCase):

    def test_gp_op_ip(self):
        layout = Cl(3)[0]
        e1 = layout.blades['e1']
        e2 = layout.blades['e2']
        e3 = layout.blades['e3']

        e123 = layout.blades['e123']
        np.testing.assert_almost_equal(e123.value, (e1 ^ e2 ^ e3).value)
        np.testing.assert_almost_equal(e123.value, (e1 * e2 * e3).value)

        e12 = layout.blades['e12']
        np.testing.assert_almost_equal(-e12.value, (e2 ^ e1).value)

        np.testing.assert_almost_equal(0, (e1 ^ e1).value)
        np.testing.assert_almost_equal(0, (e2 ^ e2).value)
        np.testing.assert_almost_equal(0, (e3 ^ e3).value)

    def test_grade_obj(self):
        algebras = [Cl(i) for i in [3, 4]] + [conformalize(Cl(3)[0])]
        for alg in algebras:
            layout = alg[0]
            for i in range(len(layout.sig)+1):
                mv = layout.randomMV()(i)
                assert i == grade_obj(mv)

class FrameTests(unittest.TestCase):

    def check_inv(self, A):
        Ainv= None
        for k in range(3):
            try:
                Ainv = A.inv
            except(ValueError):
                pass
        if Ainv ==None:
            return True        
        for m, a in enumerate(A):
            for n, b in enumerate(A.inv):
                if m == n:
                    assert(a | b == 1)
                else:
                    assert(a | b == 0)

    def test_frame_inv(self):
        for p, q in [(2, 0), (3, 0), (4, 0)]:
            layout, blades = Cl(p, q)
            A = Frame(layout.randomV(p + q))
            self.check_inv(A)

    def test_innermorphic(self):
        for p, q in [(2, 0), (3, 0), (4, 0)]:
            layout, blades = Cl(p, q)

            A = Frame(layout.randomV(p+q))
            R = layout.randomRotor()
            B = Frame([R*a*~R for a in A])
            self.assertTrue(A.is_innermorphic_to(B))



if __name__ == '__main__':
    unittest.main()
