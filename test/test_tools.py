from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
from past.builtins import range

from clifford import Cl

import unittest

from clifford.tools import orthoFrames2Verser as of2v
import numpy as np

from numpy import exp, float64, testing

from nose.plugins.skip import SkipTest

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
        """
        Checks rotation rotor generation
        """
        from clifford.tools.g3 import generate_rotation_rotor, random_unit_vector, angle_between_vectors

        euc_vector_m = random_unit_vector()
        euc_vector_n = random_unit_vector()
        theta = angle_between_vectors(euc_vector_m, euc_vector_n)
        print(theta)

        rot_rotor = generate_rotation_rotor(theta, euc_vector_m, euc_vector_n)
        v1 = euc_vector_m
        v2 = rot_rotor*euc_vector_m*~rot_rotor
        theta_return = angle_between_vectors(v1, v2)
        print(theta_return)

        testing.assert_almost_equal(theta_return, theta)
        testing.assert_almost_equal(euc_vector_n.value, v2.value)

    @SkipTest
    def test_find_rotor_aligning_vectors(self):
        """
        Currently fails, needs to be dug into
        """
        from clifford.g3c import layout
        e1 = layout.blades['e1']
        e2 = layout.blades['e2']
        from clifford.tools.g3 import random_euc_mv, random_rotation_rotor, rotor_align_vecs
        u_list = [random_euc_mv() for i in range(50)]
        for i in range(100):
            r = random_rotation_rotor()
            v_list = [r*u*~r for u in u_list]
            r_2 = rotor_align_vecs(u_list, v_list)
            print(r_2)
            print(r)
            testing.assert_almost_equal(r.value, r_2.value)



if __name__ == '__main__':
    unittest.main()
