
import pytest
import numpy as np
from numpy import testing

from clifford import Cl
from clifford.tools import orthoFrames2Versor as of2v
from clifford._numba_utils import DISABLE_JIT

from . import rng  # noqa: F401

too_slow_without_jit = pytest.mark.skipif(
    DISABLE_JIT, reason="test is too slow without JIT"
)


@pytest.mark.skip(reason="unknown")
class TestTools:

    def checkit(self, p, q, rng):  # noqa: F811
        # p, q =4,0
        N = p + q
        # eps(1e-4)
        layout, blades = Cl(p, q)

        # create frame
        A = layout.randomV(n=N, rng=rng)
        # create Rotor
        R = 5.*layout.randomRotor(rng=rng)
        # create rotated frame
        B = [R*a*~R for a in A]

        # find versor from both frames
        R_found, rs = of2v(A, B)

        # Rotor is determiend correctly, within a sign
        self.assertTrue(R == R_found or R == -R_found)

        # Determined Versor implements desired transformation
        self.assertTrue([R_found*a*~R_found for a in A] == B)

    def testOrthoFrames2VersorEuclidean(self):
        for p, q in [(2, 0), (3, 0), (4, 0)]:
            self.checkit(p=p, q=q)

    @pytest.mark.skip(reason="unknown")  # fails
    def testOrthoFrames2VersorMinkowski(self):
        for p, q in [(1, 1), (2, 1), (3, 1)]:
            self.checkit(p=p, q=q)

    @pytest.mark.skip(reason="unknown")  # fails
    def testOrthoFrames2VersorBalanced(self):
        for p, q in [(2, 2)]:
            self.checkit(p=p, q=q)


class TestG3Tools:

    def test_quaternion_conversions(self, rng):  # noqa: F811
        """
        Bidirectional rotor - quaternion test. This needs work but is a reasonable start
        """
        from clifford.tools.g3 import rotor_to_quaternion, quaternion_to_rotor
        from clifford.tools.g3c import random_rotation_rotor
        for i in range(1000):
            rotor = random_rotation_rotor(rng=rng)
            quaternion = rotor_to_quaternion(rotor)
            rotor_return = quaternion_to_rotor(quaternion)
            testing.assert_almost_equal(rotor.value, rotor_return.value)

    @too_slow_without_jit
    def test_rotation_matrix_conversions(self, rng):   # noqa: F811
        """
        Bidirectional rotor - rotation matrix test. This needs work but is a reasonable start
        """
        from clifford.g3c import down
        from clifford.tools.g3 import rotation_matrix_to_rotor, rotor_to_rotation_matrix
        from clifford.tools.g3c import random_rotation_rotor, random_conformal_point, apply_rotor
        for i in range(1000):
            rotor = random_rotation_rotor(rng=rng)
            # Check that we can map up and back
            Rmat = rotor_to_rotation_matrix(rotor)
            rotor_return = rotation_matrix_to_rotor(Rmat)

            # Check that the rotations do the same thing
            for k in range(10):
                A = random_conformal_point(rng=rng)
                B = down(apply_rotor(A, rotor)).value[1:4]
                C = Rmat @ down(A).value[1:4]
                np.testing.assert_almost_equal(B, C)

                C = down(apply_rotor(A, rotor_return)).value[1:4]
                np.testing.assert_almost_equal(B, C)

    def test_generate_rotation_rotor_and_angle(self, rng):  # noqa: F811
        """
        Checks rotation rotor generation
        """
        from clifford.tools.g3 import generate_rotation_rotor, random_unit_vector, angle_between_vectors
        for i in range(1000):
            euc_vector_m = random_unit_vector(rng=rng)
            euc_vector_n = random_unit_vector(rng=rng)
            theta = angle_between_vectors(euc_vector_m, euc_vector_n)

            rot_rotor = generate_rotation_rotor(theta, euc_vector_m, euc_vector_n)
            v1 = euc_vector_m
            v2 = rot_rotor*euc_vector_m*~rot_rotor
            theta_return = angle_between_vectors(v1, v2)

            testing.assert_almost_equal(theta_return, theta)
            testing.assert_almost_equal(euc_vector_n.value, v2.value)

    @pytest.mark.skip(reason="unknown")
    def test_find_rotor_aligning_vectors(self, rng):  # noqa: F811
        """
        Currently fails, needs to be dug into
        """
        from clifford.g3c import layout
        e1 = layout.blades['e1']
        e2 = layout.blades['e2']
        from clifford.tools.g3 import random_euc_mv, random_rotation_rotor, rotor_align_vecs
        u_list = [random_euc_mv(rng=rng) for i in range(50)]
        for i in range(100):
            r = random_rotation_rotor(rng=rng)
            v_list = [r*u*~r for u in u_list]
            r_2 = rotor_align_vecs(u_list, v_list)
            print(r_2)
            print(r)
            testing.assert_almost_equal(r.value, r_2.value)


class TestPointProcessing:
    def test_convex_hull_vertices(self, rng):  # noqa: F811
        from clifford.tools.g3c import random_conformal_point
        from clifford.tools.point_processing import GAConvexHull
        point_list = [random_conformal_point(rng=rng) for i in range(100)]
        hull = GAConvexHull(point_list, hull_dims=3)
        conf_vertices = [hull.GApoints[i] for i in hull.vertices]

        # from pyganja import GanjaScene, draw
        # gs = GanjaScene()
        # gs.add_objects(point_list, static=True, color=int('00000000', 16))
        # gs.add_objects(conf_vertices, static=True, color=int('00FF0000', 16))
        # draw(gs, scale=0.05)

    def test_convex_hull_conformal_rounds(self, rng):  # noqa: F811
        from clifford.tools.g3c import random_conformal_point
        from clifford.tools.point_processing import GAConvexHull
        point_list = [random_conformal_point(rng=rng) for i in range(100)]
        hull = GAConvexHull(point_list, hull_dims=3)
        rounds = hull.conformal_rounds()

        # from pyganja import GanjaScene, draw
        # gs = GanjaScene()
        # gs.add_objects(point_list, static=True, color=int('00000000', 16))
        # gs.add_objects(rounds, color=int('00FF0000', 16))
        # draw(gs, scale=0.05)

    def test_convex_hull_conformal_flats(self, rng):  # noqa: F811
        from clifford.tools.g3c import random_conformal_point
        from clifford.tools.point_processing import GAConvexHull

        point_list = [random_conformal_point(rng=rng) for i in range(100)]
        hull = GAConvexHull(point_list, hull_dims=3)
        flats = hull.conformal_flats()

        # from pyganja import GanjaScene, draw
        # gs = GanjaScene()
        # gs.add_objects(point_list, static=True, color=int('00000000', 16))
        # gs.add_objects(flats, color=int('00FF0000', 16))
        # draw(gs, scale=0.05)

    def test_convex_hull_facets(self, rng):  # noqa: F811
        from clifford.tools.g3c import random_conformal_point
        from clifford.tools.point_processing import GAConvexHull
        point_list = [random_conformal_point(rng=rng) for i in range(100)]
        hull = GAConvexHull(point_list, hull_dims=3)
        facets = hull.conformal_facets()

        # from pyganja import GanjaScene, draw
        # gs = GanjaScene()
        # gs.add_objects(point_list, static=True, color=int('00000000', 16))
        # for f in facets:
        #     gs.add_facet(f, color=int('AAFF0000', 16))
        # draw(gs, scale=0.05)

    def test_GADelaunay_facets(self, rng):  # noqa: F811
        from clifford.g3c import up, blades, layout
        e1 = blades['e1']
        e2 = blades['e2']
        einf = layout.einf
        from clifford.tools.g3c import random_conformal_point, project_points_to_plane
        from clifford.tools.point_processing import GADelaunay
        point_list = [random_conformal_point(rng=rng) for i in range(100)]
        point_list_flat = project_points_to_plane(point_list, (up(0)^up(e1)^up(e2)^einf).normal())
        hull = GADelaunay(point_list_flat, hull_dims=2)
        facets = hull.conformal_facets()

        # from pyganja import GanjaScene, draw
        # gs = GanjaScene()
        # gs.add_objects(point_list_flat, static=True, color=int('00000000', 16))
        # gs.add_facets(facets, static=True, color=int('00FF0000', 16))
        # draw(gs, scale=0.05)
