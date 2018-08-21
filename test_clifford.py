from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
from past.builtins import range

from clifford import Cl, randomMV, Frame, get_mult_function, conformalize, grade_obj
from clifford.tools import orthoFrames2Verser as of2v

import numpy as np
from numpy import exp, float64, testing
import unittest
import itertools

from nose.plugins.skip import SkipTest


class ConformalArrayTests(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        from clifford import g3c
        layout = g3c.layout
        self.layout = layout
        self.stuff = g3c.stuff

    def test_up_down(self):
        from clifford.tools.g3c import ConformalMVArray
        from clifford.tools.g3 import random_euc_mv

        ep, en, up, down, homo, E0, ninf, no = (self.stuff["ep"], self.stuff["en"],
                                                self.stuff["up"], self.stuff["down"], self.stuff["homo"],
                                                self.stuff["E0"], self.stuff["einf"], -self.stuff["eo"])
        mv = []
        up_mv = []
        for i in range(100):
            p = random_euc_mv()
            mv.append(p)
            up_mv.append(up(p))
        test_array = ConformalMVArray(mv)
        up_array = test_array.up()
        down_array = up_array.down()
        for a, b in zip(up_array, up_mv):
            np.testing.assert_almost_equal(a.value, b.value)
            np.testing.assert_almost_equal(a.value, b.value)
        for a, b in zip(down_array, mv):
            np.testing.assert_almost_equal(a.value, b.value)

    def test_apply_rotor(self):

        from clifford.tools.g3c import ConformalMVArray, apply_rotor
        from clifford.tools.g3 import random_euc_mv

        mv = []
        for i in range(100):
            p = random_euc_mv()
            mv.append(p)
        test_array = ConformalMVArray(mv)
        up_array = test_array.up()

        # Test apply rotor
        for i in range(100):
            R = ConformalMVArray([self.layout.randomRotor()])
            rotated_array = up_array.apply_rotor(R)
            for i, v in enumerate(rotated_array):
                np.testing.assert_almost_equal(v.value, apply_rotor(up_array[i], R[0]).value)

    def test_dual(self):
        from clifford.tools.g3c import ConformalMVArray
        from clifford.tools.g3 import random_euc_mv
        mv = []
        for i in range(100):
            p = random_euc_mv()
            mv.append(p)
        test_array = ConformalMVArray(mv)
        up_array = test_array.up()
        I5 = self.layout.blades['e12345']

        np.testing.assert_almost_equal((up_array * ConformalMVArray([I5])).value,
                                       ConformalMVArray([i * I5 for i in up_array]).value)

    def test_from_value_array(self):
        from clifford.tools.g3c import ConformalMVArray
        from clifford.tools.g3 import random_euc_mv
        mv = []
        for i in range(100):
            p = random_euc_mv()
            mv.append(p)
        test_array = ConformalMVArray(mv)
        up_array = test_array.up()
        new_mv_array = ConformalMVArray.from_value_array(up_array.value)
        np.testing.assert_almost_equal(new_mv_array.value, up_array.value)


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


class BasicConformalTests(unittest.TestCase):
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


class ModelMatchingTests(unittest.TestCase):

    def test_iterative_model_match_line_optimised(self):
        from clifford.tools.g3c import generate_random_object_cluster, \
            random_line, random_rotation_translation_rotor, apply_rotor
        from clifford.tools.g3c.model_matching import iterative_model_match

        object_generator = random_line
        n_objects_per_cluster = 20

        # Make a cluster
        cluster_objects = generate_random_object_cluster(n_objects_per_cluster, object_generator,
                                                         max_cluster_trans=0.5, max_cluster_rot=np.pi / 3)
        error_count = 0
        n_runs = 10
        for i in range(n_runs):
            # Rotate and translate the cluster
            disturbance_rotor = random_rotation_translation_rotor(maximum_translation=2, maximum_angle=np.pi/8)
            target = [apply_rotor(c, disturbance_rotor).normal() for c in cluster_objects]

            labels, costs, r_est = iterative_model_match(target, cluster_objects, 30, object_type='lines')
            try:
                assert np.sum(labels == range(n_objects_per_cluster)) == n_objects_per_cluster
            except:
                print(disturbance_rotor)
                print(r_est)
                error_count += 1
        print('Correct fraction: ', 1.0 - error_count/n_runs)

    def test_iterative_model_match(self):
        from clifford.tools.g3c import generate_random_object_cluster, \
            random_line, random_rotation_translation_rotor, apply_rotor, \
            random_circle, random_point_pair
        from clifford.tools.g3c.model_matching import iterative_model_match

        object_generator = random_line
        n_objects_per_cluster = 20

        # Make a cluster
        cluster_objects = generate_random_object_cluster(n_objects_per_cluster, object_generator,
                                                         max_cluster_trans=0.5, max_cluster_rot=np.pi / 3)
        error_count = 0
        n_runs = 10
        for i in range(n_runs):
            # Rotate and translate the cluster
            disturbance_rotor = random_rotation_translation_rotor(maximum_translation=2, maximum_angle=np.pi/8)
            target = [apply_rotor(c, disturbance_rotor).normal() for c in cluster_objects]

            labels, costs, r_est = iterative_model_match(target, cluster_objects, 30, object_type='generic')
            try:
                assert np.sum(labels == range(n_objects_per_cluster)) == n_objects_per_cluster
            except:
                print(disturbance_rotor)
                print(r_est)
                error_count += 1
        print('Correct fraction: ', 1.0 - error_count/n_runs)

    @SkipTest
    def test_iterative_model_match_cuda(self):
        from clifford.tools.g3c import generate_random_object_cluster, \
            random_line, random_rotation_translation_rotor, apply_rotor, \
            random_circle, random_point_pair
        from clifford.tools.g3c.model_matching import iterative_model_match

        object_generator = random_line
        n_objects_per_cluster = 20

        # Make a cluster
        cluster_objects = generate_random_object_cluster(n_objects_per_cluster, object_generator,
                                                         max_cluster_trans=0.5, max_cluster_rot=np.pi / 3)
        error_count = 0
        n_runs = 10
        for i in range(n_runs):
            # Rotate and translate the cluster
            disturbance_rotor = random_rotation_translation_rotor(maximum_translation=2, maximum_angle=np.pi/8)
            target = [apply_rotor(c, disturbance_rotor).normal() for c in cluster_objects]

            labels, costs, r_est = iterative_model_match(target, cluster_objects, 30,
                                                         object_type='generic', cuda=True)
            try:
                assert np.sum(labels == range(n_objects_per_cluster)) == n_objects_per_cluster
            except:
                print(disturbance_rotor)
                print(r_est)
                error_count += 1
        print('Correct fraction: ', 1.0 - error_count/n_runs)


    def test_iterative_model_match_sequential(self):
        from clifford.tools.g3c import generate_random_object_cluster, \
            random_line, random_rotation_translation_rotor, apply_rotor, \
            random_circle, random_plane
        from clifford.tools.g3c.model_matching import iterative_model_match_sequential

        object_generator = random_line
        n_objects_per_cluster = 20

        # Make a cluster
        cluster_objects = generate_random_object_cluster(n_objects_per_cluster, object_generator,
                                                         max_cluster_trans=0.5, max_cluster_rot=np.pi / 3)
        error_count = 0
        n_runs = 10
        for i in range(n_runs):
            # Rotate and translate the cluster
            disturbance_rotor = random_rotation_translation_rotor(maximum_translation=2, maximum_angle=np.pi/8)
            target = [apply_rotor(c, disturbance_rotor).normal() for c in cluster_objects]

            labels, costs, r_est = iterative_model_match_sequential(target, cluster_objects, 30, object_type='generic')
            try:
                assert np.sum(labels == range(n_objects_per_cluster)) == n_objects_per_cluster
            except:
                print(disturbance_rotor)
                print(r_est)
                error_count += 1
        print('Correct fraction: ', 1.0 - error_count/n_runs)

    @SkipTest
    def test_iterative_model_match_sequential_cuda(self):
        from clifford.tools.g3c import generate_random_object_cluster, \
            random_line, random_rotation_translation_rotor, apply_rotor, \
            random_circle, random_plane
        from clifford.tools.g3c.model_matching import iterative_model_match_sequential

        object_generator = random_line
        n_objects_per_cluster = 20

        # Make a cluster
        cluster_objects = generate_random_object_cluster(n_objects_per_cluster, object_generator,
                                                         max_cluster_trans=0.5, max_cluster_rot=np.pi / 3)
        error_count = 0
        n_runs = 10
        for i in range(n_runs):
            # Rotate and translate the cluster
            disturbance_rotor = random_rotation_translation_rotor(maximum_translation=2, maximum_angle=np.pi/8)
            target = [apply_rotor(c, disturbance_rotor).normal() for c in cluster_objects]

            labels, costs, r_est = iterative_model_match_sequential(target, cluster_objects, 30,
                                                                    object_type='generic', cuda=True)
            try:
                assert np.sum(labels == range(n_objects_per_cluster)) == n_objects_per_cluster
            except:
                print(disturbance_rotor)
                print(r_est)
                error_count += 1
        print('Correct fraction: ', 1.0 - error_count/n_runs)


    def test_iterative_model_match_newton(self):
        from clifford.tools.g3c import generate_random_object_cluster, \
            random_line, random_rotation_translation_rotor, apply_rotor
        from clifford.tools.g3c.model_matching import iterative_model_match

        object_generator = random_line
        n_objects_per_cluster = 20

        # Make a cluster
        cluster_objects = generate_random_object_cluster(n_objects_per_cluster, object_generator,
                                                         max_cluster_trans=0.5, max_cluster_rot=np.pi / 3)
        error_count = 0
        n_runs = 10
        for i in range(n_runs):
            # Rotate and translate the cluster
            disturbance_rotor = random_rotation_translation_rotor(maximum_translation=2, maximum_angle=np.pi/8)
            target = [apply_rotor(c, disturbance_rotor).normal() for c in cluster_objects]

            labels, costs, r_est = iterative_model_match(target, cluster_objects, 30, rotor_newton=True)
            try:
                assert np.sum(labels == range(n_objects_per_cluster)) == n_objects_per_cluster
            except:
                print(disturbance_rotor)
                print(r_est)
                error_count += 1
        print('Correct fraction: ', 1.0 - error_count/n_runs)

    @SkipTest
    def test_REFORM(self):
        from clifford.tools.g3c import generate_random_object_cluster, \
            random_line, random_rotation_translation_rotor, apply_rotor
        from clifford.tools.g3c.model_matching import REFORM

        object_generator = random_line
        n_objects_per_cluster = 20
        objects_per_sample = 10
        iterations = 30
        pool_size = 8

        n_samples = 8

        error_count = 0
        n_runs = 10
        for i in range(n_runs):

            # Make a cluster
            cluster_objects = generate_random_object_cluster(n_objects_per_cluster, object_generator,
                                                             max_cluster_trans=0.5, max_cluster_rot=np.pi / 3)

            # Rotate and translate the cluster
            disturbance_rotor = random_rotation_translation_rotor(maximum_translation=2, maximum_angle=np.pi/8)
            target = [apply_rotor(c, disturbance_rotor).normal() for c in cluster_objects]

            labels, costs, r_est = REFORM(target, cluster_objects, n_samples, objects_per_sample,
                                    iterations, covergence_threshold=0.00000001, pool_size=pool_size)
            try:
                assert np.sum(labels == range(n_objects_per_cluster)) == n_objects_per_cluster
            except:
                print(disturbance_rotor)
                print(r_est)
                error_count += 1
        print('Correct fraction: ', 1.0 - error_count/n_runs)

    @SkipTest
    def test_REFORM_sequential(self):
        from clifford.tools.g3c import generate_random_object_cluster, \
            random_line, random_rotation_translation_rotor, apply_rotor
        from clifford.tools.g3c.model_matching import REFORM, REFORM_sequential

        object_generator = random_line
        n_objects_per_cluster = 20
        objects_per_sample = 10
        iterations = 30
        pool_size = 8

        n_samples = 8

        error_count = 0
        n_runs = 10
        for i in range(n_runs):

            # Make a cluster
            cluster_objects = generate_random_object_cluster(n_objects_per_cluster, object_generator,
                                                             max_cluster_trans=0.5, max_cluster_rot=np.pi / 3)

            # Rotate and translate the cluster
            disturbance_rotor = random_rotation_translation_rotor(maximum_translation=2, maximum_angle=np.pi/8)
            target = [apply_rotor(c, disturbance_rotor).normal() for c in cluster_objects]

            labels, costs, r_est = REFORM_sequential(target, cluster_objects, n_samples, objects_per_sample,
                                    iterations, covergence_threshold=0.00000001, pool_size=pool_size)
            try:
                assert np.sum(labels == range(n_objects_per_cluster)) == n_objects_per_cluster
            except:
                print(disturbance_rotor)
                print(r_est)
                error_count += 1
        print('Correct fraction: ', 1.0 - error_count/n_runs)

    @SkipTest
    def test_REFORM_line_optimised(self):
        from clifford.tools.g3c import generate_random_object_cluster, \
            random_line, random_rotation_translation_rotor, apply_rotor
        from clifford.tools.g3c.model_matching import REFORM

        object_generator = random_line
        n_objects_per_cluster = 20
        objects_per_sample = 5
        iterations = 30
        pool_size = 8

        n_samples = pool_size

        error_count = 0
        n_runs = 10
        for i in range(n_runs):

            # Make a cluster
            cluster_objects = generate_random_object_cluster(n_objects_per_cluster, object_generator,
                                                             max_cluster_trans=0.5, max_cluster_rot=np.pi / 3)

            # Rotate and translate the cluster
            disturbance_rotor = random_rotation_translation_rotor(maximum_translation=2, maximum_angle=np.pi/8)
            target = [apply_rotor(c, disturbance_rotor).normal() for c in cluster_objects]

            labels, costs, r_est = REFORM(target, cluster_objects, n_samples, objects_per_sample,
                                    iterations, covergence_threshold=0.00000001, pool_size=pool_size,
                                    object_type = 'lines')
            try:
                assert np.sum(labels == range(n_objects_per_cluster)) == n_objects_per_cluster
            except:
                print(disturbance_rotor)
                print(r_est)
                error_count += 1
        print('Correct fraction: ', 1.0 - error_count/n_runs)


    @SkipTest
    def test_iterative_model_match_incomplete_query(self):
        from clifford.tools.g3c import generate_random_object_cluster, \
            random_line, random_rotation_translation_rotor, apply_rotor, \
            random_circle, random_point_pair
        from clifford.tools.g3c.model_matching import iterative_model_match, \
            iterative_model_match_sequential
        import random

        # Set the generator
        object_generator = random_line
        n_objects_per_cluster = 100
        n_keep = 50

        # Make a cluster
        cluster_objects = generate_random_object_cluster(n_objects_per_cluster, object_generator,
                                                         max_cluster_trans=0.5, max_cluster_rot=np.pi / 3)
        error_count = 0
        n_runs = 10
        for i in range(n_runs):

            # Rotate and translate the cluster
            disturbance_rotor = random_rotation_translation_rotor(maximum_translation=2, maximum_angle=np.pi/8)
            target = [apply_rotor(c, disturbance_rotor).normal() for c in cluster_objects]

            # Keep only a fixed number of the query model objects
            sample_indices = random.sample(range(n_objects_per_cluster), n_keep)
            query_model = [cluster_objects[i] for i in sample_indices]

            labels, costs, r_est = iterative_model_match(target, query_model, 30, object_type='generic')
            try:
                assert np.sum(labels == sample_indices) == n_keep
            except:
                print(disturbance_rotor)
                print(r_est)
                error_count += 1
        print('Correct fraction: ', 1.0 - error_count/n_runs)

    @SkipTest
    def test_REFORM_incomplete_query(self):
        from clifford.tools.g3c import generate_random_object_cluster, \
            random_line, random_rotation_translation_rotor, apply_rotor
        from clifford.tools.g3c.model_matching import REFORM, REFORM_sequential
        import random

        object_generator = random_line
        n_objects_per_cluster = 100
        n_keep = 50

        objects_per_sample = 10
        iterations = 30
        pool_size = 8


        n_samples = 8

        error_count = 0
        n_runs = 10
        for i in range(n_runs):

            # Make a cluster
            cluster_objects = generate_random_object_cluster(n_objects_per_cluster, object_generator,
                                                             max_cluster_trans=0.5, max_cluster_rot=np.pi / 3)

            # Rotate and translate the cluster
            disturbance_rotor = random_rotation_translation_rotor(maximum_translation=2, maximum_angle=np.pi/8)
            target = [apply_rotor(c, disturbance_rotor).normal() for c in cluster_objects]

            # Keep only a fixed number of the query model objects
            sample_indices = random.sample(range(n_objects_per_cluster), n_keep)
            query_model = [cluster_objects[i] for i in sample_indices]

            labels, costs, r_est = REFORM_sequential(target, query_model, n_samples, objects_per_sample,
                                    iterations, covergence_threshold=0.00000001, pool_size=pool_size)
            try:
                assert np.sum(labels == sample_indices) == n_keep
            except:
                print(disturbance_rotor)
                print(r_est)
                error_count += 1
        print('Correct fraction: ', 1.0 - error_count/n_runs)



class SceneSimplificationTests(unittest.TestCase):

    def test_simplify_scene(self):
        from clifford.tools.g3c import generate_n_clusters, random_line
        from clifford.tools.g3c.scene_simplification import simplify_scene
        object_generator = random_line
        n_clusters = 3
        n_objects_per_cluster = 10
        threshold = 0.5
        all_objects, object_clusters = generate_n_clusters(object_generator,
                                                           n_clusters,
                                                           n_objects_per_cluster)
        all_object_copy = [o for o in all_objects]
        all_object_copy = simplify_scene(all_object_copy, threshold)
        print(n_clusters)
        #assert len(all_object_copy) == n_clusters


class ObjectClusteringTests(unittest.TestCase):

    def run_n_clusters(self, object_generator, n_clusters, n_objects_per_cluster, n_shotgunning):
        from clifford.tools.g3c import generate_n_clusters
        from clifford.tools.g3c.object_clustering import n_clusters_objects
        all_objects, object_clusters = generate_n_clusters( object_generator, n_clusters, n_objects_per_cluster )
        [new_labels, centroids, start_labels, start_centroids] = n_clusters_objects(n_clusters, all_objects,
                                                                                    initial_centroids=None,
                                                                                    n_shotgunning=n_shotgunning,
                                                                                    averaging_method='unweighted')
        return all_objects, new_labels, centroids

    def test_clustering_point_pairs(self):
        from clifford.tools.g3c import random_point_pair
        from clifford.tools.g3c.object_clustering import visualise_n_clusters
        object_generator = random_point_pair
        n_clusters = 3
        n_objects_per_cluster = 10
        n_shotgunning = 60
        all_objects, labels, centroids = self.run_n_clusters(object_generator, n_clusters,
                                                             n_objects_per_cluster, n_shotgunning)

        sc = visualise_n_clusters(all_objects, centroids, labels, object_type='point_pair',
                             color_1=np.array([255, 0, 0]), color_2=np.array([0, 255, 0]))
        print(sc)

    def test_clustering_lines(self):
        from clifford.tools.g3c import random_line
        from clifford.tools.g3c.object_clustering import visualise_n_clusters
        object_generator = random_line
        n_clusters = 3
        n_objects_per_cluster = 10
        n_shotgunning = 60
        all_objects, labels, centroids = self.run_n_clusters(object_generator, n_clusters,
                                                             n_objects_per_cluster, n_shotgunning)
        sc = visualise_n_clusters(all_objects, centroids, labels, object_type='line',
                             color_1=np.array([255, 0, 0]), color_2=np.array([0, 255, 0]))
        print(sc)

    def test_clustering_circles(self):
        from clifford.tools.g3c import random_circle
        from clifford.tools.g3c.object_clustering import visualise_n_clusters
        object_generator = random_circle
        n_clusters = 3
        n_objects_per_cluster = 10
        n_shotgunning = 60
        all_objects, labels, centroids = self.run_n_clusters(object_generator, n_clusters,
                                                             n_objects_per_cluster, n_shotgunning)
        sc = visualise_n_clusters(all_objects, centroids, labels, object_type='circle',
                             color_1=np.array([255, 0, 0]), color_2=np.array([0, 255, 0]))
        print(sc)

    def test_clustering_spheres(self):
        from clifford.tools.g3c import random_sphere
        from clifford.tools.g3c.object_clustering import visualise_n_clusters
        object_generator = random_sphere
        n_clusters = 3
        n_objects_per_cluster = 10
        n_shotgunning = 60
        all_objects, labels, centroids = self.run_n_clusters(object_generator, n_clusters,
                                                             n_objects_per_cluster, n_shotgunning)
        sc = visualise_n_clusters(all_objects, centroids, labels, object_type='sphere',
                             color_1=np.array([255, 0, 0]), color_2=np.array([0, 255, 0]))
        print(sc)

    def test_clustering_planes(self):
        from clifford.tools.g3c import random_plane
        from clifford.tools.g3c.object_clustering import visualise_n_clusters

        object_generator = random_plane
        n_clusters = 3
        n_objects_per_cluster = 10
        n_shotgunning = 60
        all_objects, labels, centroids = self.run_n_clusters(object_generator, n_clusters,
                                                             n_objects_per_cluster, n_shotgunning)
        sc = visualise_n_clusters(all_objects, centroids, labels, object_type='plane',
                             color_1=np.array([255, 0, 0]), color_2=np.array([0, 255, 0]))
        print(sc)

    def test_assign_objects_to_objects(self):
        import numpy.testing as npt
        from clifford.tools.g3c import random_line, random_point_pair, random_plane, \
            random_circle, random_sphere, random_rotation_translation_rotor
        from clifford.tools.g3c.object_clustering import assign_measurements_to_objects_matrix

        for i in range(5):
            # point_pairs
            object_set_a = [random_point_pair() for i in range(20)]
            object_set_b = [l for l in object_set_a]
            label_a, costs_a = assign_measurements_to_objects_matrix(object_set_a, object_set_b)
            npt.assert_equal(label_a, np.array(range(len(label_a))))

            # Lines
            object_set_a = [random_line() for i in range(20)]
            object_set_b = [l for l in object_set_a]
            label_a, costs_a = assign_measurements_to_objects_matrix(object_set_a, object_set_b)
            npt.assert_equal(label_a, np.array(range(len(label_a))))

            # circles
            object_set_a = [random_circle() for i in range(20)]
            object_set_b = [l for l in object_set_a]
            label_a, costs_a = assign_measurements_to_objects_matrix(object_set_a, object_set_b)
            npt.assert_equal(label_a, np.array(range(len(label_a))))

            # planes
            object_set_a = [random_plane() for i in range(20)]
            object_set_b = [l for l in object_set_a]
            label_a, costs_a = assign_measurements_to_objects_matrix(object_set_a, object_set_b)
            npt.assert_equal(label_a, np.array(range(len(label_a))))

            # spheres
            object_set_a = [random_sphere() for i in range(20)]
            object_set_b = [l for l in object_set_a]
            label_a, costs_a = assign_measurements_to_objects_matrix(object_set_a, object_set_b)
            npt.assert_equal(label_a, np.array(range(len(label_a))))

        for i in range(10):
            r = random_rotation_translation_rotor(0.001,np.pi/32)
            # point_pairs
            object_set_a = [random_point_pair() for i in range(20)]
            object_set_b = [l for l in object_set_a]
            label_a, costs_a = assign_measurements_to_objects_matrix(object_set_a, object_set_b)
            npt.assert_equal(label_a, np.array(range(len(label_a))))

            # Lines
            object_set_a = [random_line() for i in range(20)]
            object_set_b = [l for l in object_set_a]
            label_a, costs_a = assign_measurements_to_objects_matrix(object_set_a, object_set_b)
            npt.assert_equal(label_a, np.array(range(len(label_a))))

            # circles
            object_set_a = [random_circle() for i in range(20)]
            object_set_b = [l for l in object_set_a]
            label_a, costs_a = assign_measurements_to_objects_matrix(object_set_a, object_set_b)
            npt.assert_equal(label_a, np.array(range(len(label_a))))

            # planes
            object_set_a = [random_plane() for i in range(20)]
            object_set_b = [l for l in object_set_a]
            label_a, costs_a = assign_measurements_to_objects_matrix(object_set_a, object_set_b)
            npt.assert_equal(label_a, np.array(range(len(label_a))))

            # spheres
            object_set_a = [random_sphere() for i in range(20)]
            object_set_b = [l for l in object_set_a]
            label_a, costs_a = assign_measurements_to_objects_matrix(object_set_a, object_set_b)
            npt.assert_equal(label_a, np.array(range(len(label_a))))


class RotorEstimationTests(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        from clifford import g3c
        layout = g3c.layout
        self.layout = layout
        self.stuff = g3c.stuff
        
    def run_rotor_estimation(self, object_generator, estimation_function, n_runs=5, n_objects_per_run=10):
        e1 = self.layout.blades['e1']
        e2 = self.layout.blades['e2']
        e3 = self.layout.blades['e3']
        ep, en, up, down, homo, E0, ninf, no = (self.stuff["ep"], self.stuff["en"],
                                                self.stuff["up"], self.stuff["down"], self.stuff["homo"],
                                                self.stuff["E0"], self.stuff["einf"], -self.stuff["eo"])
        from clifford.tools.g3 import generate_rotation_rotor, random_euc_mv
        from clifford.tools.g3c import generate_translation_rotor

        error_count = 0
        for i in range(n_runs):
            query_model = [object_generator().normal() for i in range(n_objects_per_run)]
            r = (generate_translation_rotor(random_euc_mv(l_max=0.01)) * generate_rotation_rotor(np.random.randn()/10,
                                                                                       random_euc_mv().normal(),
                                                                                       random_euc_mv().normal())).normal()

            reference_model = [(r * l * ~r).normal() for l in query_model]
            r_est = estimation_function(reference_model, query_model)
            error_flag = False
            for a, b in zip([(r_est * l * ~r_est).normal() for l in query_model], reference_model):
                if abs(a+b) < 0.0001:
                    c = -b
                    print('SIGN FLIP')
                else:
                    c = b
                if np.any(np.abs(a.value - c.value) > 0.01):
                    error_flag = True
            if error_flag:
                error_count += 1
            print(i, error_count)
        print('\n\nESTIMATION SUMMARY')
        print('OBJECTS ', n_objects_per_run)
        print('RUNS ', n_runs)
        print('ERRORS ', error_count)
        print('ERROR percentage ', 100*error_count/float(n_runs), '%')

    def test_estimate_rotor_lines_optimisation(self):
        from clifford.tools.g3c import random_line
        from clifford.tools.g3c.rotor_estimation import estimate_rotor_objects
        def estimation_func(pp_list_a,pp_list_b):
            r_est, costs = estimate_rotor_objects(pp_list_a, pp_list_b)
            return r_est
        self.run_rotor_estimation(random_line, estimation_func)

    def test_estimate_rotor_circles_optimisation(self):
        from clifford.tools.g3c import random_circle
        from clifford.tools.g3c.rotor_estimation import estimate_rotor_objects
        def estimation_func(pp_list_a,pp_list_b):
            r_est, costs = estimate_rotor_objects(pp_list_a, pp_list_b)
            return r_est
        self.run_rotor_estimation(random_circle, estimation_func)

    def test_estimate_rotor_point_pairs_optimisation(self):
        #""" Skip this one as it seems to take a fairly long time atm """
        from clifford.tools.g3c import random_point_pair
        from clifford.tools.g3c.rotor_estimation import estimate_rotor_objects
        def estimation_func(pp_list_a,pp_list_b):
            r_est, costs = estimate_rotor_objects(pp_list_a, pp_list_b)
            return r_est
        self.run_rotor_estimation(random_point_pair, estimation_func)

    def test_estimate_rotor_planes_optimisation(self):
        from clifford.tools.g3c import random_plane
        from clifford.tools.g3c.rotor_estimation import estimate_rotor_objects
        def estimation_func(pp_list_a,pp_list_b):
            r_est, costs = estimate_rotor_objects(pp_list_a, pp_list_b)
            return r_est
        self.run_rotor_estimation(random_plane, estimation_func)

    def test_estimate_rotor_spheres_optimisation(self):
        from clifford.tools.g3c import random_sphere
        from clifford.tools.g3c.rotor_estimation import estimate_rotor_objects
        def estimation_func(pp_list_a, pp_list_b):
            r_est, costs = estimate_rotor_objects(pp_list_a, pp_list_b)
            return r_est
        self.run_rotor_estimation(random_sphere, estimation_func)

    def test_estimate_rotor_lines_sequential(self):
        from clifford.tools.g3c import random_line
        from clifford.tools.g3c.rotor_estimation import sequential_object_rotor_estimation
        def estimation_func(pp_list_a, pp_list_b):
            r_est, r_list, exit_flag = sequential_object_rotor_estimation(pp_list_a, pp_list_b)
            print(exit_flag)
            return r_est
        self.run_rotor_estimation(random_line, estimation_func)

    def test_estimate_rotor_circles_sequential(self):
        from clifford.tools.g3c import random_circle
        from clifford.tools.g3c.rotor_estimation import sequential_object_rotor_estimation

        def estimation_func(pp_list_a, pp_list_b):
            r_est, r_list, exit_flag = sequential_object_rotor_estimation(pp_list_a, pp_list_b)
            print(exit_flag)
            return r_est
        self.run_rotor_estimation(random_circle, estimation_func)

    @SkipTest
    def test_estimate_rotor_circles_sequential_then_opt(self):
        from clifford.tools.g3c import random_circle, apply_rotor
        from clifford.tools.g3c.rotor_estimation import sequential_object_rotor_estimation, estimate_rotor_objects

        def estimation_func(pp_list_a, pp_list_b):
            r_est_1, r_list, exit_flag = sequential_object_rotor_estimation(pp_list_a, pp_list_b)
            r_est_2 = 1.0
            if exit_flag == 1:
                object_set_a = [apply_rotor(l, r_est_1).normal() for l in pp_list_a]
                r_est_2, costs = estimate_rotor_objects(object_set_a, pp_list_b)
            return r_est_2*r_est_1
        self.run_rotor_estimation(random_circle, estimation_func)

    @SkipTest
    def test_estimate_rotor_point_pairs_sequential(self):
        """ Skip this one as it seems to take a fairly long time atm """ 
        from clifford.tools.g3c import random_point_pair
        from clifford.tools.g3c.rotor_estimation import sequential_object_rotor_estimation

        def estimation_func(pp_list_a, pp_list_b):
            r_est, r_list, exit_flag = sequential_object_rotor_estimation(pp_list_a, pp_list_b)
            print(exit_flag)
            return r_est
        self.run_rotor_estimation(random_point_pair, estimation_func)

    def test_estimate_rotor_planes_sequential(self):
        from clifford.tools.g3c import random_plane
        from clifford.tools.g3c.rotor_estimation import sequential_object_rotor_estimation
        def estimation_func(pp_list_a, pp_list_b):
            r_est, r_list, exit_flag = sequential_object_rotor_estimation(pp_list_a, pp_list_b)
            print(exit_flag)
            return r_est
        self.run_rotor_estimation(random_plane, estimation_func)

    def test_estimate_rotor_spheres_sequential(self):
        from clifford.tools.g3c import random_sphere
        from clifford.tools.g3c.rotor_estimation import sequential_object_rotor_estimation
        def estimation_func(pp_list_a, pp_list_b):
            r_est, r_list, exit_flag = sequential_object_rotor_estimation(pp_list_a, pp_list_b)
            print(exit_flag)
            return r_est
        self.run_rotor_estimation(random_sphere, estimation_func)

    def test_estimate_rotor_lines_sequential_convergence_estimation(self):
        from clifford.tools.g3c import random_line
        from clifford.tools.g3c.rotor_estimation import sequential_object_rotor_estimation_convergence_detection
        def estimation_func(pp_list_a, pp_list_b):
            r_est, r_list, exit_flag = sequential_object_rotor_estimation_convergence_detection(pp_list_a, pp_list_b)
            print(exit_flag)
            return r_est
        self.run_rotor_estimation(random_line, estimation_func)

    def test_estimate_rotor_circles_sequential_convergence_estimation(self):
        from clifford.tools.g3c import random_circle
        from clifford.tools.g3c.rotor_estimation import sequential_object_rotor_estimation_convergence_detection

        def estimation_func(pp_list_a, pp_list_b):
            r_est, r_list, exit_flag = sequential_object_rotor_estimation_convergence_detection(pp_list_a, pp_list_b)
            print(exit_flag)
            return r_est
        self.run_rotor_estimation(random_circle, estimation_func)

    @SkipTest
    def test_estimate_rotor_point_pairs_sequential_convergence_estimation(self):
        """ Skip this one as it seems to take a fairly long time atm """ 
        from clifford.tools.g3c import random_point_pair
        from clifford.tools.g3c.rotor_estimation import sequential_object_rotor_estimation_convergence_detection

        def estimation_func(pp_list_a, pp_list_b):
            r_est, r_list, exit_flag = sequential_object_rotor_estimation_convergence_detection(pp_list_a, pp_list_b)
            print(exit_flag)
            return r_est
        self.run_rotor_estimation(random_point_pair, estimation_func)

    def test_estimate_rotor_planes_sequential_convergence_estimation(self):
        from clifford.tools.g3c import random_plane
        from clifford.tools.g3c.rotor_estimation import sequential_object_rotor_estimation_convergence_detection
        def estimation_func(pp_list_a, pp_list_b):
            r_est, r_list, exit_flag = sequential_object_rotor_estimation_convergence_detection(pp_list_a, pp_list_b)
            print(exit_flag)
            return r_est
        self.run_rotor_estimation(random_plane, estimation_func)

    def test_estimate_rotor_spheres_sequential_convergence_estimation(self):
        from clifford.tools.g3c import random_sphere
        from clifford.tools.g3c.rotor_estimation import sequential_object_rotor_estimation_convergence_detection
        def estimation_func(pp_list_a, pp_list_b):
            r_est, r_list, exit_flag = sequential_object_rotor_estimation_convergence_detection(pp_list_a, pp_list_b)
            print(exit_flag)
            return r_est
        self.run_rotor_estimation(random_sphere, estimation_func)

class G3CToolsTests(unittest.TestCase):


    def test_random_point_pair_at_origin(self):
        from clifford.tools.g3c import random_point_pair_at_origin
        from clifford.tools.g3c.GAOnline import GAScene

        pp_list = [random_point_pair_at_origin() for i in range(10)]

        sc = GAScene()
        for pp in pp_list:
            sc.add_point_pair(pp)
        print(sc)


    def test_random_line_at_origin(self):
        from clifford.tools.g3c import random_line_at_origin
        from clifford.tools.g3c.GAOnline import GAScene

        pp_list = [random_line_at_origin() for i in range(10)]

        sc = GAScene()
        for pp in pp_list:
            sc.add_line(pp)
        print(sc)

    def test_random_circle_at_origin(self):
        from clifford.tools.g3c import random_circle_at_origin
        from clifford.tools.g3c.GAOnline import GAScene

        pp_list = [random_circle_at_origin() for i in range(10)]

        sc = GAScene()
        for pp in pp_list:
            sc.add_circle(pp)
        print(sc)

    def test_random_sphere_at_origin(self):
        from clifford.tools.g3c import random_sphere_at_origin
        from clifford.tools.g3c.GAOnline import GAScene

        pp_list = [random_sphere_at_origin() for i in range(10)]

        sc = GAScene()
        for pp in pp_list:
            sc.add_sphere(pp)
        print(sc)

    def test_random_plane_at_origin(self):
        from clifford.tools.g3c import random_plane_at_origin
        from clifford.tools.g3c.GAOnline import GAScene

        pp_list = [random_plane_at_origin() for i in range(10)]

        sc = GAScene()
        for pp in pp_list:
            sc.add_plane(pp)
        print(sc)



    def test_generate_translation_rotor(self):
        """ Tests translation rotor generation """
        from clifford import g3c
        layout = g3c.layout
        locals().update(g3c.blades)
        ep, en, up, down, homo, E0, ninf, no = (g3c.stuff["ep"], g3c.stuff["en"],
                                                g3c.stuff["up"], g3c.stuff["down"], g3c.stuff["homo"],
                                                g3c.stuff["E0"], g3c.stuff["einf"], -g3c.stuff["eo"])
        from clifford.tools.g3 import random_euc_mv
        from clifford.tools.g3c import generate_translation_rotor

        for i in range(100):
            rand = random_euc_mv()
            starting_point = up(random_euc_mv())
            r_trans = generate_translation_rotor(rand)
            end_point = r_trans*starting_point*~r_trans
            translation_vec = down(end_point) - down(starting_point)
            testing.assert_almost_equal(translation_vec.value, rand.value)

    def test_intersect_line_and_plane_to_point(self):
        """ Intersection of a line and a plane """
        from clifford import g3c
        layout = g3c.layout
        e1 = layout.blades['e1']
        e2 = layout.blades['e2']
        e3 = layout.blades['e3']
        ep, en, up, down, homo, E0, ninf, no = (g3c.stuff["ep"], g3c.stuff["en"],
                                                g3c.stuff["up"], g3c.stuff["down"], g3c.stuff["homo"],
                                                g3c.stuff["E0"], g3c.stuff["einf"], -g3c.stuff["eo"])
        from clifford.tools.g3c import intersect_line_and_plane_to_point
        # First the case that they intersect
        line = (up(e1)^up(e1+e3)^ninf).normal()
        plane = (up(e3)^up(e3+e1)^up(e3+e2)^ninf).normal()
        point_result = intersect_line_and_plane_to_point(line, plane)
        testing.assert_almost_equal(down(point_result).value, (e3+e1).value)
        # Next the case that the do not intersect
        line = (up(0) ^ up(e1) ^ ninf).normal()
        point_result = intersect_line_and_plane_to_point(line, plane)
        assert point_result is None

    def test_normalise_n_minus_1(self):
        from clifford.tools.g3c import random_conformal_point, normalise_n_minus_1, ninf
        for i in range(500):
            mv = np.random.rand()*random_conformal_point()
            mv_normed = normalise_n_minus_1(mv)
            testing.assert_almost_equal( (mv_normed|ninf)[0], -1.0)

    @SkipTest
    def test_quaternion_and_vector_to_rotor(self):
        """
        TODO: IMPLEMENT THIS TEST
        """
        # quaternion_and_vector_to_rotor(quaternion, vector)

    def test_get_properties_of_sphere(self):
        from clifford import g3c
        layout = g3c.layout
        e1 = layout.blades['e1']
        e2 = layout.blades['e2']
        e3 = layout.blades['e3']
        ep, en, up, down, homo, E0, ninf, no = (g3c.stuff["ep"], g3c.stuff["en"],
                                                g3c.stuff["up"], g3c.stuff["down"], g3c.stuff["homo"],
                                                g3c.stuff["E0"], g3c.stuff["einf"], -g3c.stuff["eo"])
        from clifford.tools.g3 import random_euc_mv
        from clifford.tools.g3c import get_radius_from_sphere, get_center_from_sphere, \
            generate_translation_rotor

        for i in range(100):
            # Make a sphere
            scale_factor = np.random.rand()
            sphere = (up(scale_factor*e1)^up(-scale_factor*e1)^up(scale_factor*e3)^up(scale_factor*e2)).normal()
            # Translate it
            rand_trans = random_euc_mv()
            trans_rot = generate_translation_rotor(rand_trans)
            sphere = (trans_rot*sphere*~trans_rot).normal()

            center = get_center_from_sphere(sphere)
            radius = get_radius_from_sphere(sphere)

            testing.assert_almost_equal(down(center).value, rand_trans.value)
            testing.assert_almost_equal(radius, scale_factor)

    def test_point_pair_to_end_points(self):
        from clifford.tools.g3c import random_conformal_point, point_pair_to_end_points
        for i in range(100):
            point_a = random_conformal_point()
            point_b = random_conformal_point()
            pp = (point_a^point_b).normal()
            p_a, p_b = point_pair_to_end_points(pp)
            testing.assert_almost_equal(p_a.value, point_a.value)
            testing.assert_almost_equal(p_b.value, point_b.value)

    def test_euc_distance(self):
        from clifford import g3c
        layout = g3c.layout
        e1 = layout.blades['e1']
        e2 = layout.blades['e2']
        e3 = layout.blades['e3']
        ep, en, up, down, homo, E0, ninf, no = (g3c.stuff["ep"], g3c.stuff["en"],
                                                g3c.stuff["up"], g3c.stuff["down"], g3c.stuff["homo"],
                                                g3c.stuff["E0"], g3c.stuff["einf"], -g3c.stuff["eo"])
        from clifford.tools.g3c import random_conformal_point, euc_dist
        for i in range(100):
            point_a = random_conformal_point()
            point_b = random_conformal_point()
            dist = euc_dist(point_a, point_b)
            dist_alt = float(abs(down(point_a) - down(point_b)))
            testing.assert_almost_equal(dist, dist_alt)

    def test_dilation_rotor(self):
        from clifford.tools.g3c import random_sphere, generate_dilation_rotor, get_radius_from_sphere
        for i in range(100):
            scale = 2*np.random.rand()
            r = generate_dilation_rotor(scale)
            sphere = random_sphere()
            radius = get_radius_from_sphere(sphere)
            sphere2 = (r*sphere*~r).normal()
            radius2 = get_radius_from_sphere(sphere2)
            testing.assert_almost_equal(scale, radius2/radius)

    def test_rotor_between_objects(self):
        from clifford import g3c
        layout = g3c.layout
        e1 = layout.blades['e1']
        e2 = layout.blades['e2']
        e3 = layout.blades['e3']
        ep, en, up, down, homo, E0, ninf, no = (g3c.stuff["ep"], g3c.stuff["en"],
                                                g3c.stuff["up"], g3c.stuff["down"], g3c.stuff["homo"],
                                                g3c.stuff["E0"], g3c.stuff["einf"], -g3c.stuff["eo"])
        from clifford.tools.g3c import random_sphere, \
            random_plane, random_line, random_circle, \
            random_point_pair, rotor_between_objects

        for i in range(600):
            if i < 100:
                C1 = random_sphere()
                C2 = random_sphere()
            elif i < 200:
                C1 = random_plane()
                C2 = random_plane()
            elif i < 300:
                C1 = random_line()
                C2 = random_line()
            elif i < 400:
                C1 = random_circle()
                C2 = random_circle()
            elif i < 500:
                C1 = random_point_pair()
                C2 = random_point_pair()
            R = rotor_between_objects(C1, C2)
            C3 = (R*C1*~R).normal()
            # NOTE this sign check should not be used in an ideal world, need something a bit better
            if abs(C3 + C2) < 0.0001:
                print('SIGN FLIP')
                C3 = -C3
            testing.assert_almost_equal(C2.value, C3.value)


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
