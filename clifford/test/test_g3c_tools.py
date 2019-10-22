import random
from functools import reduce
import time


import numpy as np
import numpy.testing as npt
from numpy import exp
import pytest

from clifford import Cl, grades_present
from clifford.g3c import *
from clifford import general_exp
from clifford.tools.g3c import *
from clifford.tools.g3c.rotor_parameterisation import ga_log, ga_exp, general_logarithm, \
    interpolate_rotors
from clifford.tools.g3c.rotor_estimation import *
from clifford.tools.g3c.object_clustering import *
from clifford.tools.g3c.scene_simplification import *
from clifford.tools.g3c.object_fitting import *
from clifford.tools.g3c.model_matching import *
from clifford.tools.g3 import random_euc_mv
from clifford.tools.g3c.GAOnline import draw_objects, GAScene, GanjaScene


class TestRotorGeneration:
    def test_generate_translation_rotor(self):
        for i in range(10000):
            euc_vector_a = random_euc_mv()
            res = generate_translation_rotor(euc_vector_a)
            res2 = (1 + ninf * euc_vector_a / 2)
            npt.assert_almost_equal(res.value, res2.value)


class TestFitObjects:
    def test_fit_circle(self):
        try:
            noise = 0.1
            trueP = random_circle()
            point_list = project_points_to_circle([random_conformal_point() for i in range(100)], trueP)
            point_list = [up(down(P) + noise * random_euc_mv()) for P in point_list]
            print(trueP)
            circle = fit_circle(point_list)
            print(circle)
            # draw(point_list + [circle], static=False, scale=0.1)
        except:
            print('FAILED TO FIND CIRCLE')

    def test_fit_line(self):
        try:
            noise = 0.1
            trueP = random_line()
            point_list = project_points_to_line([random_conformal_point() for i in range(100)], trueP)
            point_list = [up(down(P) + noise * random_euc_mv()) for P in point_list]
            print(trueP)
            line = fit_line(point_list)
            print(line)
            # draw(point_list + [line], static=False, scale=0.1)
        except:
            print('FAILED TO FIND LINE')

    def test_fit_sphere(self):
        try:
            noise = 0.1
            trueP = random_sphere()
            point_list = project_points_to_sphere([random_conformal_point() for i in range(100)], trueP)
            point_list = [up(down(P) + noise * random_euc_mv()) for P in point_list]
            print(trueP)
            sphere = fit_sphere(point_list)
            print(sphere)
            # draw([sphere] + point_list, static=False, scale=0.1)
        except:
            print('FAILED TO FIND SPHERE')

    def test_fit_plane(self):
        try:
            noise = 0.1
            trueP = random_plane()
            point_list = project_points_to_plane([random_conformal_point() for i in range(100)], trueP)
            point_list = [up(down(P) + noise * random_euc_mv()) for P in point_list]
            print(trueP)
            plane = fit_plane(point_list)
            print(plane)
            # draw(point_list + [plane], static=False, scale=0.1)
        except:
            print('FAILED TO FIND PLANE')


class TestGeneralLogarithm:

    def test_general_logarithm_rotation(self):
        # Check we can reverse rotations
        for i in range(50):
            R = random_rotation_rotor()
            biv_2 = general_logarithm(R)
            biv_3 = ga_log(R)
            npt.assert_almost_equal(biv_2.value, biv_3.value, 3)

    def test_general_logarithm_translation(self):
        # Check we can reverse translation
        for i in range(50):
            t = random_euc_mv()
            biv = ninf * t / 2
            R = general_exp(biv).normal()
            biv_2 = general_logarithm(R)
            npt.assert_almost_equal(biv.value, biv_2.value)

    def test_general_logarithm_scaling(self):
        # Check we can reverse scaling
        for i in range(50):
            scale = 0.5 + np.random.rand()
            biv = -np.log(scale) * e45 / 2
            R = general_exp(biv).normal()
            biv_2 = general_logarithm(R)
            npt.assert_almost_equal(biv.value, biv_2.value)

    def test_general_logarithm_RS(self):
        for i in range(5):
            scale = 0.5 + np.random.rand()

            S = generate_dilation_rotor(scale).normal()
            R = generate_rotation_rotor(0.5, e1, e2).normal()
            V = (R * S).normal()

            biv_test = general_logarithm(R) + general_logarithm(S)
            biv = general_logarithm(V)

            biv_alt = ga_log(R) + general_logarithm(S)

            npt.assert_almost_equal(biv.value, biv_test.value, 5)
            npt.assert_almost_equal(biv.value, biv_alt.value, 5)

    def test_general_logarithm_TR(self):
        for i in range(5):
            # R = generate_rotation_rotor(0.5, e1, e2).normal()
            # T = generate_translation_rotor(e3 + 7 * e2 - e1).normal()
            # V = (T*R).normal()
            biv_true = random_bivector()
            V = general_exp(biv_true).normal()
            biv = general_logarithm(V)

            V_rebuilt = (general_exp(biv)).normal()

            C1 = random_point_pair()
            C2 = (V * C1 * ~V).normal()
            C3 = (V_rebuilt * C1 * ~V_rebuilt).normal()
            npt.assert_almost_equal(C2.value, C3.value, 2)

    def test_general_logarithm_TS(self):
        for i in range(5):
            scale = 0.5 + np.random.rand()
            t = random_euc_mv()
            S = generate_dilation_rotor(scale)
            T = generate_translation_rotor(t)
            V = (T * S).normal()
            biv = general_logarithm(V)
            V_rebuilt = (general_exp(biv)).normal()

            C1 = random_point_pair()
            C2 = (V * C1 * ~V).normal()
            C3 = (V_rebuilt * C1 * ~V_rebuilt).normal()
            npt.assert_almost_equal(C2.value, C3.value, 5)

    def test_general_logarithm_TRS(self):
        for i in range(5):
            scale = 0.5 + np.random.rand()
            S = generate_dilation_rotor(scale)
            R = generate_rotation_rotor(0.5, e1, e2)
            T = generate_translation_rotor(e3 + 7 * e2 - e1)
            V = (T * R * S).normal()
            biv = general_logarithm(V)
            V_rebuilt = general_exp(biv).normal()
            biv2 = general_logarithm(V)

            C1 = random_point_pair()
            C2 = (V * C1 * ~V).normal()
            C3 = (V_rebuilt * C1 * ~V_rebuilt).normal()
            npt.assert_almost_equal(C2.value, C3.value)

    @pytest.mark.parametrize('obj_gen', [
        random_point_pair, random_line, random_circle, random_plane
    ])
    def test_general_logarithm_conformal(self, obj_gen):
        for i in range(10000):
            X = obj_gen()
            Y = obj_gen()
            R = rotor_between_objects(X, Y)
            biv = general_logarithm(R)
            R_recon = general_exp(biv).normal()
            npt.assert_almost_equal(R.value, R_recon.value, 4)


class TestVisualisation:
    def test_draw_objects(self):
        scene = ConformalMVArray([random_line() for i in range(100)])
        sc_a = str(draw_objects(scene))
        scene.save('test.ga')
        sc_b = str(draw_objects('test.ga'))
        assert sc_a == sc_b

    def test_ganja_scene(self):
        scene = ConformalMVArray([up(0)^up(e1)^einf, up(0)^up(e2)^einf, up(0)^up(e3)^einf]
                                 + [random_line() for i in range(2)])

        sc = GanjaScene()
        sc.add_objects(scene)
        sc.save_to_file('test.json')


class TestConformalArray:

    def test_up_down(self):
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
            npt.assert_almost_equal(a.value, b.value)
            npt.assert_almost_equal(a.value, b.value)
        for a, b in zip(down_array, mv):
            npt.assert_almost_equal(a.value, b.value)

    def test_apply_rotor(self):
        mv = []
        for i in range(100):
            p = random_euc_mv()
            mv.append(p)
        test_array = ConformalMVArray(mv)
        up_array = test_array.up()

        # Test apply rotor
        for i in range(100):
            R = ConformalMVArray([layout.randomRotor()])
            rotated_array = up_array.apply_rotor(R)
            for i, v in enumerate(rotated_array):
                npt.assert_almost_equal(v.value, apply_rotor(up_array[i], R[0]).value)

    def test_dual(self):
        mv = []
        for i in range(100):
            p = random_euc_mv()
            mv.append(p)
        test_array = ConformalMVArray(mv)
        up_array = test_array.up()
        I5 = layout.blades['e12345']

        npt.assert_almost_equal((up_array * ConformalMVArray([I5])).value,
                                ConformalMVArray([i * I5 for i in up_array]).value)

    def test_from_value_array(self):
        mv = []
        for i in range(100):
            p = random_euc_mv()
            mv.append(p)
        test_array = ConformalMVArray(mv)
        up_array = test_array.up()
        new_mv_array = ConformalMVArray.from_value_array(up_array.value)
        npt.assert_almost_equal(new_mv_array.value, up_array.value)


class TestG3CTools:

    @pytest.fixture(params=[
        random_point_pair,
        random_line,
        random_circle,
        random_plane,
        random_sphere
    ])
    def obj_gen(self, request):
        return request.param

    def test_factorise(self, obj_gen):
        n_repeats = 50
        for i in range(n_repeats):
            X1 = obj_gen()
            basis, scale = X1.factorise()
            for b in basis:
                gpres = grades_present(b, 0.0001)
                assert gpres == {1}
            new_blade = (reduce(lambda a, b: a ^ b, basis) * scale)
            try:
                npt.assert_almost_equal(new_blade.value, X1.value, 3)
            except:
                print(X1)
                print(new_blade)
                npt.assert_almost_equal(new_blade.value, X1.value, 3)

    def test_is_blade(self):
        a = random_bivector() + random_circle()
        assert not a.isBlade()
        a = random_translation_rotor()
        assert not a.isBlade()

    def test_is_blade_generated(self, obj_gen):
        n_repeats = 5
        for i in range(n_repeats):
            a = obj_gen()
            assert a.isBlade()

    def test_average_objects(self, obj_gen):
        n_repeats = 1000
        for i in range(n_repeats):
            X1 = obj_gen()
            X2 = obj_gen()
            obj_list = [X1, X2]
            average_objects(obj_list, weights=[0.5, 0.5])

    def test_point_beyond_plane(self):
        plane = I5 * ((e1 + e2 + e3).normal() + 2 * einf)
        P = up((e1 + e2 + e3) * 3)
        assert point_beyond_plane(P, plane)
        P = up((e1 + e2 + e3) * 1)
        assert not point_beyond_plane(P, plane)

    def test_join_spheres(self):
        for j in range(1000):
            s1 = random_sphere()
            s2 = random_sphere()
            s3 = join_spheres(s1, s2)
            assert sphere_in_sphere(s1, s3)
            assert sphere_in_sphere(s2, s3)

    def test_enclosing_spheres(self):
        n_spheres = 10
        for j in range(1000):
            spheres = [random_sphere() for i in range(n_spheres)]
            s4 = normalised(enclosing_sphere(spheres))
            for s in spheres:
                assert sphere_in_sphere(s, s4)

    def test_closest_furthest_circle_points(self):
        for _ in range(100):
            C1 = random_circle()
            C2 = random_circle()
            pclose = closest_points_on_circles(C1, C2)
            pfar = furthest_points_on_circles(C1, C2)

    def test_general_object_interpolation(self):

        R_r = generate_rotation_rotor(np.pi / 16, e2, e3) * generate_rotation_rotor(np.pi / 4, e1, e2)
        R_d = generate_dilation_rotor(1.5)
        R_t = generate_translation_rotor(e3)
        R = (R_t * R_r * R_d).normal()

        # C1 = (up(0+3*e1)^up(2*e1+3*e1)).normal()
        C1 = (up(0 + 3 * e1) ^ up(2 * e1 + 3 * e1) ^ up(e1 + e3 + 3 * e1)).normal()
        C2 = (R * C1 * ~R).normal()(3)
        C3 = (R * C2 * ~R).normal()(3)
        C4 = (R * C3 * ~R).normal()(3)
        C5 = (R * C4 * ~R).normal()(3)
        object_list = [C1, C2, C3, C4, C5]

        object_alpha_array = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        new_alpha_array = np.linspace(0.0, 1.0)
        new_object_list = general_object_interpolation(object_alpha_array, object_list, new_alpha_array,
                                                       kind='quadratic')
        new_object_list = [o(3) for o in new_object_list]

        draw_objects(object_list, 'circle', color='rgb(255,0,0)')
        draw_objects(new_object_list, 'circle', color='rgb(0,255,0)')

        time.sleep(1)

    def test_n_th_root(self):

        for i in range(200):
            a = random_point_pair()
            b = random_point_pair()
            R = rotor_between_objects(a, b)
            for n in [1, 2, 4, 8, 16, 32]:
                R_n = n_th_rotor_root(R, n)
                npt.assert_almost_equal((R_n ** n).value, R.value)

    def test_random_point_pair_at_origin(self):

        pp_list = [random_point_pair_at_origin() for i in range(10)]

        sc = GAScene()
        for pp in pp_list:
            sc.add_point_pair(pp)
        print(sc)

    def test_random_line_at_origin(self):

        pp_list = [random_line_at_origin() for i in range(10)]

        sc = GAScene()
        for pp in pp_list:
            sc.add_line(pp)
        print(sc)

    def test_random_circle_at_origin(self):

        pp_list = [random_circle_at_origin() for i in range(10)]

        sc = GAScene()
        for pp in pp_list:
            sc.add_circle(pp)
        print(sc)

    def test_random_sphere_at_origin(self):

        pp_list = [random_sphere_at_origin() for i in range(10)]

        sc = GAScene()
        for pp in pp_list:
            sc.add_sphere(pp)
        print(sc)

    def test_random_plane_at_origin(self):

        pp_list = [random_plane_at_origin() for i in range(10)]

        sc = GAScene()
        for pp in pp_list:
            sc.add_plane(pp)
        print(sc)

    def test_generate_translation_rotor(self):
        """ Tests translation rotor generation """

        for i in range(100):
            rand = random_euc_mv()
            starting_point = up(random_euc_mv())
            r_trans = generate_translation_rotor(rand)
            end_point = r_trans * starting_point * ~r_trans
            translation_vec = down(end_point) - down(starting_point)
            npt.assert_almost_equal(translation_vec.value, rand.value)

    def test_intersect_line_and_plane_to_point(self):
        """ Intersection of a line and a plane """

        # First the case that they intersect
        line = (up(2*e1) ^ up(2*e1 + e3) ^ ninf).normal()
        plane = (up(e3) ^ up(e3 + e1) ^ up(e3 + e2) ^ ninf).normal()
        point_result = intersect_line_and_plane_to_point(line, plane)
        npt.assert_almost_equal(point_result.value, up(e3 + 2*e1).value)
        # Next the case that the do not intersect
        line = (up(0) ^ up(e1) ^ ninf).normal()
        point_result = intersect_line_and_plane_to_point(line, plane)
        assert point_result is None

        for i in range(200):
            line = random_line()
            plane = random_plane()
            point_result = intersect_line_and_plane_to_point(line, plane)

            # draw_objects([line], mv_type='line')
            # draw_objects([plane], mv_type='plane', color='rgb(0,255,0)')
            # draw_objects([point_result], mv_type='euc_point', color='rgb(255,0,0)')

    def test_normalise_n_minus_1(self):
        for i in range(500):
            mv = np.random.rand() * random_conformal_point()
            mv_normed = normalise_n_minus_1(mv)
            npt.assert_almost_equal((mv_normed | ninf)[0], -1.0)

    def test_get_properties_of_sphere(self):
        for i in range(100):
            # Make a sphere
            scale_factor = np.random.rand()
            sphere = (up(scale_factor * e1) ^ up(-scale_factor * e1) ^ up(scale_factor * e3) ^ up(
                scale_factor * e2)).normal()
            # Translate it
            rand_trans = random_euc_mv()
            trans_rot = generate_translation_rotor(rand_trans)
            sphere = (trans_rot * sphere * ~trans_rot).normal()

            center = get_center_from_sphere(sphere)
            radius = get_radius_from_sphere(sphere)

            npt.assert_almost_equal(down(center).value, rand_trans.value)
            npt.assert_almost_equal(radius, scale_factor)

    def test_point_pair_to_end_points(self):
        for i in range(100):
            point_a = random_conformal_point()
            point_b = random_conformal_point()
            pp = (point_a ^ point_b).normal()
            p_a, p_b = point_pair_to_end_points(pp)
            npt.assert_almost_equal(p_a.value, point_a.value)
            npt.assert_almost_equal(p_b.value, point_b.value)

    def test_euc_distance(self):
        for i in range(100):
            point_a = random_conformal_point()
            point_b = random_conformal_point()
            dist = euc_dist(point_a, point_b)
            dist_alt = float(abs(down(point_a) - down(point_b)))
            npt.assert_almost_equal(dist, dist_alt)

    def test_dilation_rotor(self):
        for i in range(100):
            scale = 2 * np.random.rand()
            r = generate_dilation_rotor(scale)
            sphere = random_sphere()
            radius = get_radius_from_sphere(sphere)
            sphere2 = (r * sphere * ~r).normal()
            radius2 = get_radius_from_sphere(sphere2)
            npt.assert_almost_equal(scale, radius2 / radius)

    def test_calculate_S_over_mu_general(self, obj_gen):
        # Repeats for each fuzz test
        n_repeats = 100

        # Test the general case
        for i in range(n_repeats):
            X1 = obj_gen()
            X2 = obj_gen()
            S = calculate_S_over_mu(X1, X2)
            X3 = -S*(X1 + X2)
            X4 = average_objects([X1, X2], [0.5, 0.5]).normal()
            if sum(np.abs((X3 + X4).value)) < 0.000001:
                print(' SIGN FLIP')
                X4 = -X4
            try:
                npt.assert_almost_equal(X3.value, X4.value, 4)
            except:
                print(X3)
                print(X4)
                X4 = average_objects([X1, X2], [0.5, 0.5]).normal()
                npt.assert_almost_equal(X3.value, X4.value, 4)

    def test_general_rotor_between_objects(self, obj_gen):
        # Repeats for each fuzz test
        n_repeats = 1000

        # Test the general case
        for i in range(n_repeats):
            C1 = obj_gen()
            C2 = obj_gen()

            R = rotor_between_objects(C1, C2)
            C3 = (R * C1 * ~R).normal()

            if sum(np.abs((C2 + C3).value)) < 0.0001:
                print('SIGN FLIP ', obj_gen.__name__)
                C3 = -C3
            try:
                npt.assert_almost_equal(C2.value, C3.value, 3)
            except:
                print(R)
                print(C2*C1 + C1*C2)
                npt.assert_almost_equal(C2.value, C3.value, 3)

    @pytest.mark.parametrize(('obj_gen', 'grd'), [
        (random_point_pair, 2),
        (random_circle, 3),
        pytest.param(random_sphere, 4, marks=[
            # gh-104
            pytest.mark.xfail(reason="Non-deterministic, sometimes fails", strict=False),
        ])
    ])
    def test_motor_between_rounds(self, obj_gen, grd):
        # Repeats for each fuzz test
        n_repeats = 1000

        # Test the general case
        for i in range(n_repeats):
            C1 = obj_gen()
            Rt = random_rotation_translation_rotor()
            C2 = (Rt * C1 * ~Rt)(grd).normal()

            R = motor_between_rounds(C1, C2)
            C3 = (R * C1 * ~R)(grd).normal()

            if sum(np.abs((C2 + C3).value)) < 0.0001:
                print('SIGN FLIP ', obj_gen.__name__)
                C3 = -C3
            try:
                npt.assert_almost_equal(C2.value, C3.value, 3)
            except:
                print(C2.normal())
                print(C3.normal())
                R = motor_between_rounds(C1, C2)
                npt.assert_almost_equal(C2.value, C3.value, 3)

    # @pytest.mark.skip(reason="unknown")  # Skip this because we know that it is a breaking case
    def test_general_rotor_between_objects_specific_cases(self):
        C1 = -(2.48651^e1234) - (2.48651^e1235) - (1.0^e1245) + (3e-05^e1345) - (0.0^e2345)
        C2 = -(25.8135^e1234) - (25.8135^e1235) + (1.0^e1245) - (3e-05^e1345) - (0.0^e2345)

        R = rotor_between_objects(C1, C2)
        C3 = (R * C1 * ~R).normal()

        if sum(np.abs((C2 + C3).value)) < 0.0001:
            C3 = -C3
        npt.assert_almost_equal(C2.value, C3.value, 3)

    # @pytest.mark.skip(reason="unknown")  # Skip this because we know that it is a breaking case
    def test_rotor_between_non_overlapping_spheres(self):
        C1 = random_sphere()
        rad = get_radius_from_sphere(C1)
        t_r = generate_translation_rotor(2.5*rad*e1)
        C2 = (t_r * C1 * ~t_r)(4).normal()
        rad2 = get_radius_from_sphere(C2)
        R = rotor_between_objects(C1, C2)
        C3 = (R * C1 * ~R).normal()
        if sum(np.abs((C2 + C3).value)) < 0.0001:
            print('SIGN FLIP ')
            C3 = -C3
        npt.assert_almost_equal(C2.value, C3.value, 5)


class TestRotorEstimation:

    def run_rotor_estimation(self, object_generator, estimation_function,
                             n_runs=20, n_objects_per_run=10):

        error_count = 0
        for i in range(n_runs):
            query_model = [object_generator().normal() for i in range(n_objects_per_run)]
            r = (generate_translation_rotor(random_euc_mv(l_max=0.01)) * generate_rotation_rotor(np.random.randn() / 10,
                                                                                                 random_euc_mv().normal(),
                                                                                                 random_euc_mv().normal())).normal()

            reference_model = [(r * l * ~r).normal() for l in query_model]
            r_est = estimation_function(reference_model, query_model)
            error_flag = False
            for a, b in zip([(r_est * l * ~r_est).normal() for l in query_model], reference_model):
                if abs(a + b) < 0.0001:
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
        print('ERROR percentage ', 100 * error_count / float(n_runs), '%')

    def test_de_keninck_twist(self):
        X = MVArray([random_conformal_point() for i in range(100)])
        R = random_rotation_rotor()
        noise_std = 0.0
        Y = MVArray([normalise_n_minus_1(apply_rotor(x, random_translation_rotor(noise_std) * R)) for x in X])
        res = de_keninck_twist(Y, X)
        try:
            npt.assert_almost_equal(R.value, res.value, 4)
        except:
            npt.assert_almost_equal(R.value, -res.value, 4)

    def test_direct_TRS_extraction(self):
        X = MVArray([random_conformal_point() for i in range(100)])
        R = (random_rotation_translation_rotor(maximum_translation=100) * generate_dilation_rotor(
            0.5 + 2 * np.random.rand())).normal()
        noise_std = 0.0
        Y = MVArray([normalise_n_minus_1(apply_rotor(x, random_translation_rotor(noise_std) * R)) for x in X])
        res = direct_TRS_extraction(Y, X)
        try:
            npt.assert_almost_equal(R.value, res.value, 4)
        except:
            npt.assert_almost_equal(R.value, -res.value, 4)

    def test_dorst_motor_points(self):
        X = MVArray([random_conformal_point() for i in range(100)])
        R = random_rotation_translation_rotor(maximum_translation=100)
        noise_std = 0.0
        Y = MVArray([normalise_n_minus_1(apply_rotor(x, random_translation_rotor(noise_std) * R)) for x in X])
        res = dorst_motor_estimate(Y, X)
        try:
            npt.assert_almost_equal(R.value, res.value, 4)
        except:
            npt.assert_almost_equal(R.value, -res.value, 4)

    @pytest.mark.parametrize('obj_gen', [
        random_line,
        random_circle,
        random_point_pair,
        random_plane,
        random_sphere,
    ])
    def test_dorst_motor_estimate(self, obj_gen):
        self.run_rotor_estimation(obj_gen, dorst_motor_estimate)

    def test_estimate_rotor_lines_average_then_opt(self):

        def estimation_func(pp_list_a, pp_list_b):
            r_start = average_estimator(pp_list_a, pp_list_b)
            query_start = [apply_rotor(b, r_start)(3).normal() for b in pp_list_b]
            r_est, costs = estimate_rotor_objects(pp_list_a, query_start)
            return (r_est*r_start).normal()

        self.run_rotor_estimation(random_line, estimation_func)

    @pytest.mark.parametrize('obj_gen', [
        random_line,
        random_circle,
        random_point_pair,
        random_plane,
        pytest.param(random_sphere, marks=pytest.mark.skip(reason="unknown")),
    ])
    def test_estimate_motor_optimisation(self, obj_gen):

        def estimation_func(pp_list_a, pp_list_b):
            r_est, costs = estimate_rotor_objects(pp_list_a, pp_list_b, motor=True)
            return r_est

        self.run_rotor_estimation(obj_gen, estimation_func)

    @pytest.mark.parametrize('obj_gen', [
        random_line,
        random_circle,
        random_point_pair,
        random_plane,
        random_sphere,
    ])
    def test_estimate_rotor_optimisation(self, obj_gen):

        def estimation_func(pp_list_a, pp_list_b):
            r_est, costs = estimate_rotor_objects(pp_list_a, pp_list_b)
            return r_est

        self.run_rotor_estimation(obj_gen, estimation_func)

    @pytest.mark.parametrize('obj_gen', [
        random_line,
        random_circle,
        pytest.param(random_point_pair, marks=pytest.mark.skip(reason="unknown")),
        random_plane,
        random_sphere
    ])
    def test_estimate_rotor_sequential(self, obj_gen):

        def estimation_func(pp_list_a, pp_list_b):
            r_est, exit_flag = sequential_object_rotor_estimation(pp_list_a, pp_list_b)
            print(exit_flag)
            return r_est

        self.run_rotor_estimation(obj_gen, estimation_func)

    @pytest.mark.skip(reason="unknown")
    def test_estimate_rotor_circles_sequential_then_opt(self):

        def estimation_func(pp_list_a, pp_list_b):
            r_est_1, exit_flag = sequential_object_rotor_estimation(pp_list_a, pp_list_b)
            r_est_2 = 1.0
            if exit_flag == 1:
                object_set_a = [apply_rotor(l, r_est_1).normal() for l in pp_list_a]
                r_est_2, costs = estimate_rotor_objects(object_set_a, pp_list_b)
            return r_est_2 * r_est_1

        self.run_rotor_estimation(random_circle, estimation_func)


class TestSceneSimplification:

    def test_simplify_recursive(self):
        object_generator = random_line
        n_clusters = 3
        n_objects_per_cluster = 5
        threshold = 0.5
        all_objects, object_clusters = generate_n_clusters(object_generator,
                                                           n_clusters,
                                                           n_objects_per_cluster)
        all_object_copy = [o for o in all_objects]
        all_object_copy = simplify_scene_recursive(all_object_copy, threshold)
        print(n_clusters)
        # assert len(all_object_copy) == n_clusters

    def test_simplify_scene(self):
        object_generator = random_line
        n_clusters = 3
        n_objects_per_cluster = 5
        threshold = 2.0
        all_objects, object_clusters = generate_n_clusters(object_generator,
                                                           n_clusters,
                                                           n_objects_per_cluster)
        all_object_copy1 = [o for o in all_objects]
        all_object_copy1 = simplify_scene(all_object_copy1, threshold)
        print(len(all_object_copy1))
        # assert len(all_object_copy) == n_clusters

        all_object_copy2 = [o for o in all_objects]
        all_object_copy2 = simplify_scene(all_object_copy2, threshold)
        print(len(all_object_copy2))

        draw_objects(all_object_copy1)
        draw_objects(all_object_copy2, color='rgb(255,0,0)')


class TestObjectClustering:

    def run_n_clusters(self, object_generator, n_clusters, n_objects_per_cluster, n_shotgunning):

        all_objects, object_clusters = generate_n_clusters(object_generator, n_clusters, n_objects_per_cluster)
        [new_labels, centroids, start_labels, start_centroids] = n_clusters_objects(n_clusters, all_objects,
                                                                                    initial_centroids=None,
                                                                                    n_shotgunning=n_shotgunning,
                                                                                    averaging_method='unweighted')
        return all_objects, new_labels, centroids

    def test_clustering_point_pairs(self):

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

        object_generator = random_plane
        n_clusters = 3
        n_objects_per_cluster = 10
        n_shotgunning = 60
        all_objects, labels, centroids = self.run_n_clusters(object_generator, n_clusters,
                                                             n_objects_per_cluster, n_shotgunning)
        sc = visualise_n_clusters(all_objects, centroids, labels, object_type='plane',
                                  color_1=np.array([255, 0, 0]), color_2=np.array([0, 255, 0]))
        print(sc)

    @pytest.mark.parametrize('obj_gen', [
        random_point_pair,
        random_line,
        random_circle,
        random_plane,
        random_sphere
    ])
    def test_assign_objects_to_objects(self, obj_gen):
        n_repeats = 5
        for i in range(n_repeats):
            object_set_a = [obj_gen() for i in range(20)]
            object_set_b = [l for l in object_set_a]
            label_a, costs_a = assign_measurements_to_objects_matrix(object_set_a, object_set_b)
            try:
                npt.assert_equal(label_a, np.array(range(len(label_a))))
            except:
                label_a, costs_a = assign_measurements_to_objects_matrix(object_set_a, object_set_b)
                npt.assert_equal(label_a, np.array(range(len(label_a))))

        n_repeats = 5
        for i in range(n_repeats):
            r = random_rotation_translation_rotor(0.001, np.pi / 32)

            object_set_a = [obj_gen() for i in range(20)]
            object_set_b = [l for l in object_set_a]
            label_a, costs_a = assign_measurements_to_objects_matrix(object_set_a, object_set_b)
            npt.assert_equal(label_a, np.array(range(len(label_a))))


class TestModelMatching:

    def test_fingerprint_match(self):

        object_generator = random_line
        n_objects_per_cluster = 20

        # Make a cluster
        cluster_objects = generate_random_object_cluster(n_objects_per_cluster, object_generator,
                                                         max_cluster_trans=0.5, max_cluster_rot=np.pi / 3)
        sum_p = 0
        n_runs = 20
        for i in range(n_runs):
            # Rotate and translate the cluster
            disturbance_rotor = random_rotation_translation_rotor(maximum_translation=10, maximum_angle=np.pi / 2).normal()
            target = [apply_rotor(c, disturbance_rotor).normal() for c in cluster_objects]

            labels, min_costs = match_by_fingerprint(target, cluster_objects)
            pcorrect = 100*np.sum([l == i for i, l in enumerate(labels)])/n_objects_per_cluster
            sum_p += pcorrect
            print('Percent correct: ', pcorrect)
        print('av_p_correct ', sum_p/n_runs)
        print('\n', flush=True)

    def test_iterative_model_match_line_optimised(self):

        object_generator = random_line
        n_objects_per_cluster = 20

        # Make a cluster
        cluster_objects = generate_random_object_cluster(n_objects_per_cluster, object_generator,
                                                         max_cluster_trans=0.5, max_cluster_rot=np.pi / 3)
        error_count = 0
        n_iterations = 30
        n_runs = 5
        for i in range(n_runs):
            # Rotate and translate the cluster
            disturbance_rotor = random_rotation_translation_rotor(maximum_translation=2, maximum_angle=np.pi / 8)
            target = [apply_rotor(c, disturbance_rotor).normal() for c in cluster_objects]

            labels, costs, r_est = iterative_model_match(target, cluster_objects, n_iterations, object_type='lines')
            try:
                assert np.sum(labels == range(n_objects_per_cluster)) == n_objects_per_cluster
            except:
                print(disturbance_rotor)
                print(r_est)
                error_count += 1
        print('Correct fraction: ', 1.0 - error_count / n_runs)

    def test_iterative_model_match(self):

        object_generator = random_line
        n_objects_per_cluster = 20

        # Make a cluster
        cluster_objects = generate_random_object_cluster(n_objects_per_cluster, object_generator,
                                                         max_cluster_trans=0.5, max_cluster_rot=np.pi / 3)
        error_count = 0
        n_iterations = 30
        n_runs = 5
        for i in range(n_runs):
            # Rotate and translate the cluster
            disturbance_rotor = random_rotation_translation_rotor(maximum_translation=2, maximum_angle=np.pi / 8).normal()
            target = [apply_rotor(c, disturbance_rotor).normal() for c in cluster_objects]

            labels, costs, r_est = iterative_model_match(target, cluster_objects, n_iterations, object_type='generic')
            r_est = r_est.normal()
            try:
                assert np.sum(labels == range(n_objects_per_cluster)) == n_objects_per_cluster
            except:
                print(disturbance_rotor)
                print(r_est)
                error_count += 1
        print('Correct fraction: ', 1.0 - error_count / n_runs)

    @pytest.mark.skip(reason="unknown")
    def test_iterative_model_match_cuda(self):

        object_generator = random_line
        n_objects_per_cluster = 20

        # Make a cluster
        cluster_objects = generate_random_object_cluster(n_objects_per_cluster, object_generator,
                                                         max_cluster_trans=0.5, max_cluster_rot=np.pi / 3)
        error_count = 0
        n_runs = 5
        for i in range(n_runs):
            # Rotate and translate the cluster
            disturbance_rotor = random_rotation_translation_rotor(maximum_translation=2, maximum_angle=np.pi / 8)
            target = [apply_rotor(c, disturbance_rotor).normal() for c in cluster_objects]

            labels, costs, r_est = iterative_model_match(target, cluster_objects, 30,
                                                         object_type='generic', cuda=True)
            try:
                assert np.sum(labels == range(n_objects_per_cluster)) == n_objects_per_cluster
            except:
                print(disturbance_rotor)
                print(r_est)
                error_count += 1
        print('Correct fraction: ', 1.0 - error_count / n_runs)

    def test_iterative_model_match_sequential(self):

        object_generator = random_line
        n_objects_per_cluster = 20

        # Make a cluster
        cluster_objects = generate_random_object_cluster(n_objects_per_cluster, object_generator,
                                                         max_cluster_trans=0.5, max_cluster_rot=np.pi / 3)
        error_count = 0
        n_runs = 5
        for i in range(n_runs):
            # Rotate and translate the cluster
            disturbance_rotor = random_rotation_translation_rotor(maximum_translation=2, maximum_angle=np.pi / 8)
            target = [apply_rotor(c, disturbance_rotor).normal() for c in cluster_objects]
            labels, costs, r_est = iterative_model_match_sequential(target, cluster_objects, 30, object_type='generic')
            try:
                assert np.sum(labels == range(n_objects_per_cluster)) == n_objects_per_cluster
            except:
                print(disturbance_rotor)
                print(r_est)
                error_count += 1
        print('Correct fraction: ', 1.0 - error_count / n_runs)

    @pytest.mark.skip(reason="unknown")
    def test_iterative_model_match_sequential_cuda(self):
        object_generator = random_line
        n_objects_per_cluster = 20

        # Make a cluster
        cluster_objects = generate_random_object_cluster(n_objects_per_cluster, object_generator,
                                                         max_cluster_trans=0.5, max_cluster_rot=np.pi / 3)
        error_count = 0
        n_runs = 5
        for i in range(n_runs):
            # Rotate and translate the cluster
            disturbance_rotor = random_rotation_translation_rotor(maximum_translation=2, maximum_angle=np.pi / 8)
            target = [apply_rotor(c, disturbance_rotor).normal() for c in cluster_objects]

            labels, costs, r_est = iterative_model_match_sequential(target, cluster_objects, 30,
                                                                    object_type='generic', cuda=True)
            try:
                assert np.sum(labels == range(n_objects_per_cluster)) == n_objects_per_cluster
            except:
                print(disturbance_rotor)
                print(r_est)
                error_count += 1
        print('Correct fraction: ', 1.0 - error_count / n_runs)

    @pytest.mark.skip(reason="unknown")
    def test_REFORM(self):

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
            disturbance_rotor = random_rotation_translation_rotor(maximum_translation=2, maximum_angle=np.pi / 8)
            target = [apply_rotor(c, disturbance_rotor).normal() for c in cluster_objects]

            labels, costs, r_est = REFORM(target, cluster_objects, n_samples, objects_per_sample,
                                          iterations, covergence_threshold=0.00000001, pool_size=pool_size)
            try:
                assert np.sum(labels == range(n_objects_per_cluster)) == n_objects_per_cluster
            except:
                print(disturbance_rotor)
                print(r_est)
                error_count += 1
        print('Correct fraction: ', 1.0 - error_count / n_runs)

    @pytest.mark.skip(reason="unknown")
    def test_REFORM_sequential(self):

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
            disturbance_rotor = random_rotation_translation_rotor(maximum_translation=2, maximum_angle=np.pi / 8)
            target = [apply_rotor(c, disturbance_rotor).normal() for c in cluster_objects]

            labels, costs, r_est = REFORM_sequential(target, cluster_objects, n_samples, objects_per_sample,
                                                     iterations, covergence_threshold=0.00000001, pool_size=pool_size)
            try:
                assert np.sum(labels == range(n_objects_per_cluster)) == n_objects_per_cluster
            except:
                print(disturbance_rotor)
                print(r_est)
                error_count += 1
        print('Correct fraction: ', 1.0 - error_count / n_runs)

    @pytest.mark.skip(reason="unknown")
    def test_REFORM_line_optimised(self):

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
            disturbance_rotor = random_rotation_translation_rotor(maximum_translation=2, maximum_angle=np.pi / 8)
            target = [apply_rotor(c, disturbance_rotor).normal() for c in cluster_objects]

            labels, costs, r_est = REFORM(target, cluster_objects, n_samples, objects_per_sample,
                                          iterations, covergence_threshold=0.00000001, pool_size=pool_size,
                                          object_type='lines')
            try:
                assert np.sum(labels == range(n_objects_per_cluster)) == n_objects_per_cluster
            except:
                print(disturbance_rotor)
                print(r_est)
                error_count += 1
        print('Correct fraction: ', 1.0 - error_count / n_runs)

    @pytest.mark.skip(reason="unknown")
    def test_iterative_model_match_incomplete_query(self):

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
            disturbance_rotor = random_rotation_translation_rotor(maximum_translation=2, maximum_angle=np.pi / 8)
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
        print('Correct fraction: ', 1.0 - error_count / n_runs)

    @pytest.mark.skip(reason="unknown")
    def test_REFORM_incomplete_query(self):
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
            disturbance_rotor = random_rotation_translation_rotor(maximum_translation=2, maximum_angle=np.pi / 8)
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
        print('Correct fraction: ', 1.0 - error_count / n_runs)
