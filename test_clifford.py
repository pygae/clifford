from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
from past.builtins import range

from clifford import Cl, randomMV, Frame
from clifford.tools import orthoFrames2Verser as of2v

from numpy import exp, float64
import unittest

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

    def test_add_float64(self):
        '''
        test array_wrap method to take control addition from numpy array
        '''
        layout, blades = self.algebras[0]
        e1 = blades['e1']

        float64(1) + e1
        self.assertEqual(1 + e1, float64(1) + e1)


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


@SkipTest
class ToolsTests(unittest.TestCase):

    def checkit(self, p, q):
        # p, q =4,0
        N = p + q
        # eps(1e-4)
        layout, blades = Cl(p, q)

        # create frame
        A = layout.randomV(n=N)
        # create Rotor
        R = 5.*layout.randomRotor()
        # create rotated frame
        B = [R*a*~R for a in A]

        # find verser from both frames
        R_found, rs = of2v(A, B)

        # Rotor is determiend correctly, within a sign
        self.assertTrue(R == R_found or R == -R_found)

        # Determined Verser implements desired transformation
        self.assertTrue([R_found*a*~R_found for a in A] == B)

    def testOrthoFrames2VerserEuclidean(self):
        for p, q in [(2, 0), (3, 0), (4, 0)]:
            self.checkit(p=p, q=q)

    @SkipTest  # fails
    def testOrthoFrames2VerserMinkowski(self):
        for p, q in [(1, 1), (2, 1), (3, 1)]:
            self.checkit(p=p, q=q)

    @SkipTest  # fails
    def testOrthoFrames2VerserBalanced(self):
        for p, q in [(2, 2)]:
            self.checkit(p=p, q=q)


if __name__ == '__main__':
    unittest.main()
