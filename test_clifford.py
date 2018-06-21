from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
from past.builtins import range

from clifford import Cl, randomMV, Frame, get_mult_function, conformalize, grade_obj
from clifford.tools import orthoFrames2Verser as of2v

from numpy import exp, float64, testing
import unittest
import itertools

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


class BasicAlgebraTests(unittest.TestCase):

    def test_grade_obj(self):
        algebras = [Cl(i) for i in [3, 4]] + [conformalize(Cl(3)[0])]
        for alg in algebras:
            layout = alg[0]
            for i in range(len(layout.sig)+1):
                mv = layout.randomMV()(i)
                assert i == grade_obj(mv)

    def test_sparse_multiply(self):
        algebras = [Cl(i) for i in [3, 4]] + [conformalize(Cl(3)[0])]
        # For all the algebras we are interested in
        for alg in algebras:
            layout = alg[0]
            # Make two random multivectors
            a = layout.randomMV()
            b = layout.randomMV()
            # Project the multivectors to the grades required
            grades_possibilities = []
            for r in range(1,len(layout.sig)):
                possible_grades = [list(m) for m in list(itertools.combinations(range(len(layout.sig)), r))]
                grades_possibilities += possible_grades
            for i,grades_a in enumerate(grades_possibilities):
                sparse_mv_a = sum([a(k) for k in grades_a])
                for j,grades_b in enumerate(grades_possibilities):
                    sparse_mv_b = sum([b(k) for k in grades_b])
                    # Compute results
                    gp = get_mult_function(layout.gmt,layout.gaDims,layout.gradeList,grades_a=grades_a,grades_b=grades_b)
                    result_sparse = gp(sparse_mv_a.value,sparse_mv_b.value)
                    result_dense = (sparse_mv_a*sparse_mv_b).value
                    # Check they are the same
                    testing.assert_almost_equal(result_sparse, result_dense)
                    print(j+i*len(grades_possibilities),len(grades_possibilities)**2)


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



class G3ToolsTests(unittest.TestCase):

    def test_quaternion_conversions(self):
        """
        Bidirectional rotor - quaternion test. This needs work but is a reasonable start
        """
        from clifford.g3c import layout
        from clifford.tools.g3 import rotor_to_quaternion, quaternion_to_rotor
        e1 = layout.blades['e1']
        e2 = layout.blades['e2']
        rotor = e1*e2
        print(rotor)
        quaternion = rotor_to_quaternion(rotor)
        print(quaternion)
        rotor_return = quaternion_to_rotor(quaternion)
        print(rotor_return)
        testing.assert_almost_equal(rotor.value, rotor_return.value)


    def test_rotation_matrix_conversions(self):
        """
        Bidirectional rotor - rotation matrix test. This needs work but is a reasonable start
        """
        from clifford.g3c import layout
        from clifford.tools.g3 import rotation_matrix_to_rotor, rotor_to_rotation_matrix
        e1 = layout.blades['e1']
        e2 = layout.blades['e2']

        rotor = e1*e2
        print(rotor)
        matrix = rotor_to_rotation_matrix(rotor)
        print(matrix)
        rotor_return = rotation_matrix_to_rotor(matrix)
        print(rotor_return)
        testing.assert_almost_equal(rotor.value, rotor_return.value)

    def test_generate_rotation_rotor_and_angle(self):
        import numpy as np
        theta = np.random.randn()
        euc_vector_m = np.random.randn(3)
        generate_rotation_rotor(theta, euc_vector_m, euc_vector_n)



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
