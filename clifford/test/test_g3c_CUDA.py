import numpy as np
from numpy import exp, float64, testing
import numba.cuda
import unittest
import itertools


import math
import time
import clifford as cf
from clifford.tools.g3c import *
from clifford.tools.g3c.cuda import *
from clifford import g3c
from clifford.tools.g3c.cost_functions import val_object_cost_function, \
    object_set_cost_matrix

from clifford.tools.g3c.model_matching import REFORM_cuda, iterative_model_match_sequential, \
    assign_measurements_to_objects_matrix

from clifford.tools.g3c.rotor_estimation import sequential_object_rotor_estimation


object_generators = [random_point_pair, random_line, random_circle,
                     random_sphere, random_plane]


@unittest.skipIf(
    not numba.cuda.is_available(),
    "CUDA not available")
class CUDATESTS(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        layout = g3c.layout
        self.layout = layout
        self.stuff = g3c.stuff
    #
    #
    # def test_sequential_rotor_estimation_jit(self):
    #     n_mvs = 1000
    #     query_model = [random_line() for i in range(n_mvs)]
    #
    #     r = random_rotation_translation_rotor()
    #
    #     reference_model = [(r*a*~r).normal() for a in query_model]
    #
    #     query_model_array = np.array(query_model)
    #     reference_model_array = np.array(reference_model)
    #
    #     n_samples = 100
    #     n_objects_per_sample = 100
    #     output = np.zeros((n_samples, 32))
    #     mv_d_array = np.zeros(output.shape)
    #
    #     cost_array = np.zeros(n_samples, dtype=np.float64)
    #
    #     print('Starting kernel')
    #     t = time.time()
    #
    #     output, cost_array = sequential_rotor_estimation_chunks(reference_model_array, query_model_array, n_samples,
    #                                               n_objects_per_sample)
    #     end_time = time.time() - t
    #     print('Kernel finished')
    #     print(end_time)
    #
    #     # Now do the non cuda kernel
    #     t = time.time()
    #     for i in range(output.shape[0]):
    #         mv_d_array[i, :] = sequential_object_rotor_estimation(reference_model,
    #                                                                                     query_model)[0].value
    #     print(time.time() - t)
    #     print(cost_array)
    #     np.testing.assert_almost_equal(output, mv_d_array, 5)
    #

    def test_sequential_rotor_estimation_kernel(self):
        n_mvs = 1000
        query_model = [random_line() for i in range(n_mvs)]

        r = random_rotation_translation_rotor()

        reference_model = [(r*a*~r).normal() for a in query_model]

        query_model_array = np.array(query_model)
        reference_model_array = np.array(reference_model)

        n_samples = 100
        n_objects_per_sample = 100
        output = np.zeros((n_samples, 32))
        mv_d_array = np.zeros(output.shape)

        print('Starting kernel')
        t = time.time()

        output, cost_array = sequential_rotor_estimation_cuda(reference_model_array, query_model_array, n_samples,
                                                              n_objects_per_sample)
        end_time = time.time() - t
        print('Kernel finished')
        print(end_time)

        # Now do the non cuda kernel
        t = time.time()
        for i in range(output.shape[0]):
            mv_d_array[i, :] = sequential_object_rotor_estimation(reference_model,
                                                                  query_model)[0].value
        print(time.time() - t)
        print(cost_array)
        np.testing.assert_almost_equal(output, mv_d_array, 5)

    def test_square_root_of_rotor_kernel(self):
        n_mvs = 500
        mv_a_list = [random_line() for i in range(n_mvs)]
        mv_b_list = [random_line() for i in range(n_mvs)]
        rotor_list = [rotor_between_objects(C1, C2) for C1, C2 in zip(mv_a_list, mv_b_list)]
        rotor_list_array = np.array([mv.value for mv in rotor_list])
        rotor_root_array = np.zeros(rotor_list_array.shape)
        mv_d_array = np.zeros(rotor_list_array.shape)

        print('Starting kernel')
        t = time.time()
        blockdim = 64
        griddim = int(math.ceil(n_mvs / blockdim))
        square_root_of_rotor_kernel[griddim, blockdim](rotor_list_array, rotor_root_array)
        end_time = time.time() - t
        print('Kernel finished')
        print(end_time)

        # Now do the non cuda kernel
        t = time.time()
        for i in range(rotor_list_array.shape[0]):
            mv_d_array[i, :] = square_roots_of_rotor(rotor_list[i])[0].value
        print(time.time() - t)

        np.testing.assert_almost_equal(rotor_root_array, mv_d_array)

    def test_rotor_between_lines(self):
        # Make a big array of data
        n_mvs = 1000
        mv_a_array = np.array([random_line().value for i in range(n_mvs)])
        mv_b_array = np.array([random_line().value for i in range(n_mvs)])

        mv_c_array = np.zeros(mv_b_array.shape)
        mv_d_array = np.zeros(mv_b_array.shape)

        print('Starting kernel')
        t = time.time()
        blockdim = 64
        griddim = int(math.ceil(n_mvs / blockdim))
        rotor_between_lines_kernel[griddim, blockdim](mv_a_array, mv_b_array, mv_c_array)
        end_time = time.time() - t
        print('Kernel finished')
        print(end_time)

        # Now do the non cuda kernel
        t = time.time()
        for i in range(mv_a_array.shape[0]):
            mv_d_array[i, :] = val_rotor_between_lines(mv_a_array[i, :], mv_b_array[i, :])
        print(time.time() - t)

        np.testing.assert_almost_equal(mv_c_array, mv_d_array)

    def test_normalise_mvs_kernel(self):

        n_mvs = 500
        mv_a_array = np.pi*np.array([random_line().value for i in range(n_mvs)])
        mv_d_array = np.zeros(mv_a_array.shape)
        mv_b_array = mv_a_array.copy()

        print('Starting kernel')
        t = time.time()
        blockdim = 64
        griddim = int(math.ceil(n_mvs / blockdim))
        normalise_mvs_kernel[griddim, blockdim](mv_a_array)
        end_time = time.time() - t
        print('Kernel finished')
        print(end_time)

        # Now do the non cuda kernel
        t = time.time()
        for i in range(mv_a_array.shape[0]):
            mv_a = cf.MultiVector(self.layout, mv_b_array[i, :])
            mv_d_array[i, :] = mv_a.normal().value
        print(time.time() - t)

        np.testing.assert_almost_equal(mv_a_array, mv_d_array)

    def test_rotor_between_objects(self):
        # Make a big array of data
        n_mvs = 1000
        generator_list = [random_point_pair, random_line, random_circle,
                          random_sphere, random_plane]
        for generator in generator_list:
            mv_a_array = np.array([generator().value for i in range(n_mvs)], dtype=np.double)
            mv_b_array = np.array([generator().value for i in range(n_mvs)], dtype=np.double)

            mv_c_array = np.zeros(mv_b_array.shape, dtype=np.double)
            mv_d_array = np.zeros(mv_b_array.shape, dtype=np.double)

            print('Starting kernel')
            t = time.time()
            blockdim = 64
            griddim = int(math.ceil(n_mvs / blockdim))
            rotor_between_objects_kernel[griddim, blockdim](mv_a_array, mv_b_array, mv_c_array)
            end_time = time.time() - t
            print('Kernel finished')
            print(end_time)

            # Now do the non cuda kernel
            t = time.time()
            for i in range(mv_a_array.shape[0]):
                mv_a = cf.MultiVector(self.layout, mv_a_array[i, :])
                mv_b = cf.MultiVector(self.layout, mv_b_array[i, :])
                mv_d_array[i, :] = rotor_between_objects(mv_a, mv_b).value
            print(time.time() - t)
            print(generator.__name__)
            np.testing.assert_almost_equal(mv_c_array, mv_d_array)

    def test_dorst_norm_val(self):

        # Make a big array of data
        n_mvs = 500
        mv_a_list = [random_line() for i in range(n_mvs)]
        mv_b_list = [random_line() for i in range(n_mvs)]

        c_list = [1 + b*a for a, b in zip(mv_a_list, mv_b_list)]
        sigma_list = [c*~c for c in c_list]

        mv_sigma_array = np.array(sigma_list, dtype=np.double)
        mv_c_array = np.zeros(mv_sigma_array.shape[0], dtype=np.double)
        mv_d_array = np.zeros(mv_sigma_array.shape[0], dtype=np.double)

        print('Starting kernel')
        t = time.time()
        blockdim = 64
        griddim = int(math.ceil(n_mvs / blockdim))
        dorst_norm_val_kernel[griddim, blockdim](mv_sigma_array, mv_c_array)
        end_time = time.time() - t
        print('Kernel finished')
        print(end_time)

        # Now do the non cuda kernel
        t = time.time()
        for i in range(len(mv_a_list)):
            sigma = sigma_list[i]
            k_value = dorst_norm_val(sigma.value)
            mv_d_array[i] = k_value
        print(time.time() - t)
        np.testing.assert_almost_equal(mv_c_array, mv_d_array)

    def test_gp(self):

        n_mvs = 500
        mv_a_array = np.array([self.layout.randomMV().value for i in range(n_mvs)])
        mv_b_array = np.array([self.layout.randomMV().value for i in range(n_mvs)])

        mv_c_array = np.zeros(mv_b_array.shape)
        mv_d_array = np.zeros(mv_b_array.shape)

        print('Starting kernel')
        t = time.time()
        blockdim = 64
        griddim = int(math.ceil(n_mvs/blockdim))
        gp_kernel[griddim, blockdim](mv_a_array, mv_b_array, mv_c_array)
        end_time = time.time() - t
        print('Kernel finished')
        print(end_time)

        # Now do the non cuda kernel
        t = time.time()
        for i in range(mv_a_array.shape[0]):
            mv_d_array[i, :] = self.layout.gmt_func(mv_a_array[i, :], mv_b_array[i, :])
        print(time.time() - t)

        np.testing.assert_almost_equal(mv_c_array, mv_d_array)

    def test_ip(self):

        n_mvs = 500
        mv_a_array = np.array([self.layout.randomMV().value for i in range(n_mvs)])
        mv_b_array = np.array([self.layout.randomMV().value for i in range(n_mvs)])

        mv_c_array = np.zeros(mv_b_array.shape)
        mv_d_array = np.zeros(mv_b_array.shape)

        print('Starting kernel')
        t = time.time()
        blockdim = 64
        griddim = int(math.ceil(n_mvs / blockdim))
        ip_kernel[griddim, blockdim](mv_a_array, mv_b_array, mv_c_array)
        end_time = time.time() - t
        print('Kernel finished')
        print(end_time)

        # Now do the non cuda kernel
        t = time.time()
        for i in range(mv_a_array.shape[0]):
            mv_d_array[i, :] = self.layout.imt_func(mv_a_array[i, :], mv_b_array[i, :])
        print(time.time() - t)

        np.testing.assert_almost_equal(mv_c_array, mv_d_array)

    def test_apply_rotor_kernel(self):
        n_mvs = 1000
        for obj_gen in object_generators:
            mv_s = [obj_gen() for i in range(n_mvs)]
            mv_a_array = np.array([j.value for j in mv_s])
            random_rotors = [random_rotation_translation_rotor()*generate_dilation_rotor(np.random.rand()*3) for i in range(n_mvs)]
            rotor_array = np.array([mv.value for mv in random_rotors])
            output = np.zeros(rotor_array.shape)

            blockdim = 64
            griddim = int(math.ceil(n_mvs / blockdim))
            apply_rotor_kernel[griddim, blockdim](mv_a_array, rotor_array, output)

            output_normal = np.array([apply_rotor(mv, r).value for mv, r in zip(mv_s, random_rotors)])
            for i in range(n_mvs):
                try:
                    np.testing.assert_almost_equal(output_normal[i, :], output[i, :])
                except AssertionError:
                    print(output_normal[i, :])
                    print(output[i, :])
                    np.testing.assert_almost_equal(output_normal[i, :], output[i, :])

    def test_adjoint(self):

        n_mvs = 1000
        mv_a_array = np.array([self.layout.randomMV().value for i in range(n_mvs)])
        mv_d_array = np.zeros(mv_a_array.shape)
        mv_c_array = np.zeros(mv_a_array.shape)
        blockdim = 64
        griddim = int(math.ceil(n_mvs / blockdim))
        adjoint_kernel[griddim, blockdim](mv_a_array, mv_c_array)
        for i in range(mv_a_array.shape[0]):
            mv_d_array[i, :] = self.layout.adjoint_func(mv_a_array[i, :])

        np.testing.assert_almost_equal(mv_c_array, mv_d_array, 5)

    def test_rotor_cost(self):

        # Make a big array of data
        n_mvs = 500
        mv_a_array = np.array([random_line().value for i in range(n_mvs)])
        mv_b_array = np.array([random_line().value for i in range(n_mvs)])

        mv_c_array = np.zeros(n_mvs)
        mv_d_array = np.zeros(n_mvs)

        # Multiply together a load of them and see how long it takes

        print('Starting kernel')
        t = time.time()
        blockdim = 64
        griddim = int(math.ceil(n_mvs / blockdim))
        cost_line_to_line_kernel[griddim, blockdim](mv_a_array, mv_b_array, mv_c_array)
        end_time = time.time() - t
        print('Kernel finished')
        print(end_time)

        # Now do the non cuda kernel
        t = time.time()
        for i in range(mv_a_array.shape[0]):
            mv_d_array[i] = val_object_cost_function(mv_a_array[i, :], mv_b_array[i, :])
        print(time.time() - t)

        np.testing.assert_almost_equal(mv_c_array, mv_d_array, 5)

    def test_line_set_cost(self):
        n_mvs = 50
        mv_a_array = [random_line() for i in range(n_mvs)]
        mv_b_array = [random_line() for i in range(n_mvs)]
        print(mv_a_array)
        print('Starting kernel')
        t = time.time()
        mv_c_array = line_set_cost_cuda_mvs(mv_a_array, mv_b_array)
        end_time = time.time() - t
        print('Kernel finished')
        print(end_time)

        t = time.time()
        mv_d_array = object_set_cost_matrix(mv_a_array, mv_b_array, object_type='generic')
        print(time.time() - t)
        try:
            np.testing.assert_almost_equal(mv_c_array, mv_d_array, 3)
        except AssertionError:
            print(mv_c_array[0, :])
            print(mv_d_array[0, :])
            np.testing.assert_almost_equal(mv_c_array, mv_d_array, 3)

    def test_object_set_cost(self):
        n_mvs = 100
        generator_list = [random_point_pair, random_line, random_circle,
                          random_sphere, random_plane]
        for generator in generator_list:
            mv_a_array = [generator() for i in range(n_mvs)]
            mv_b_array = [generator() for i in range(n_mvs)]
            print(mv_a_array)
            print('Starting kernel')
            t = time.time()
            mv_c_array = object_set_cost_cuda_mvs(mv_a_array, mv_b_array)
            end_time = time.time() - t
            print('Kernel finished')
            print(end_time)

            t = time.time()
            mv_d_array = object_set_cost_matrix(mv_a_array, mv_b_array, object_type='generic')
            print(time.time() - t)

            for i in range(n_mvs):
                for j in range(n_mvs):
                    try:
                        assert abs(mv_d_array[i, j]-mv_c_array[i, j])/abs(mv_d_array[i, j]) < 10**(-6)
                    except AssertionError:
                        print(generator.__name__)
                        if generator.__name__ == 'random_line':
                            print(val_object_cost_function(mv_a_array[i].value, mv_b_array[j].value))
                        print(mv_d_array[i, j])
                        print(mv_c_array[i, j])
                        print(abs(mv_d_array[i, j]-mv_c_array[i, j])/abs(mv_d_array[i, j]))
                        assert abs(mv_d_array[i, j]-mv_c_array[i, j])/abs(mv_d_array[i, j]) < 10**(-6)

    def test_REFORM_cuda(self):

        object_generator = random_line

        n_objects_per_cluster = 20
        objects_per_sample = 10
        n_samples = 100
        iterations = 100

        n_runs = 5

        error_count = 0
        for i in range(n_runs):

            # Make a cluster
            cluster_objects = generate_random_object_cluster(n_objects_per_cluster, object_generator,
                                                             max_cluster_trans=0.5, max_cluster_rot=np.pi / 3)

            # Rotate and translate the cluster
            disturbance_rotor = random_rotation_translation_rotor(maximum_translation=2, maximum_angle=np.pi / 8)
            target = [apply_rotor(c, disturbance_rotor).normal() for c in cluster_objects]

            labels, costs, r_est = REFORM_cuda(target, cluster_objects, n_samples, objects_per_sample,
                                               iterations, mutation_probability=None)
            try:
                assert np.sum(labels == list(range(n_objects_per_cluster))) == n_objects_per_cluster
            except AssertionError:
                print(disturbance_rotor)
                print(r_est)
                error_count += 1
        print('Correct fraction: ', 1.0 - error_count / n_runs)

    def test_assign_objects_to_objects_cuda(self):
        n_repeats = 5
        for obj_gen in object_generators:
            print(obj_gen.__name__)
            for i in range(n_repeats):
                object_set_a = [obj_gen() for i in range(20)]
                object_set_b = [l for l in object_set_a]
                label_a, costs_a = assign_measurements_to_objects_matrix(object_set_a, object_set_b, cuda=True)
                try:
                    np.testing.assert_equal(label_a, np.array(range(len(label_a))))
                except AssertionError:
                    label_a, costs_a = assign_measurements_to_objects_matrix(object_set_a, object_set_b, cuda=True)
                    np.testing.assert_equal(label_a, np.array(range(len(label_a))))

        n_repeats = 5
        for obj_gen in object_generators:
            print(obj_gen.__name__)
            for i in range(n_repeats):
                r = random_rotation_translation_rotor(0.001, np.pi / 32)

                object_set_a = [obj_gen() for i in range(20)]
                object_set_b = [l for l in object_set_a]
                label_a, costs_a = assign_measurements_to_objects_matrix(object_set_a, object_set_b, cuda=True)
                np.testing.assert_equal(label_a, np.array(range(len(label_a))))
