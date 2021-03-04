"""
All basic test identities come from:

Easter, R.B., Hitzer, E. Double Conformal Geometric Algebra.
Adv. Appl. Clifford Algebras 27, 2175–2199 (2017).
https://doi.org/10.1007/s00006-017-0784-0
"""

import pytest
import numpy as np

from clifford._numba_utils import DISABLE_JIT

from . import rng  # noqa: F401

too_slow_without_jit = pytest.mark.skipif(
    DISABLE_JIT, reason="test is too slow without JIT"
)


def setup_module():
    # do this separately so that we get distinct timing information for it
    import clifford.dg3c  # noqa: F401


@too_slow_without_jit
class TestBasic:
    def test_metric(self):
        """
        Ensure that the metric comes out with a double copy of
        the CGA metric
        """
        from clifford.dg3c import layout

        assert np.all(layout.metric == np.array([
            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., -1., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., -1.]]))

    def test_up_down(self, rng):  # noqa: F811
        """
        Test that we can map points up and down into the dpga
        """
        from clifford.dg3c import up, down

        for i in range(1 if DISABLE_JIT else 100):
            pnt_vector = rng.standard_normal(3)
            pnt = up(pnt_vector)
            res = down(100*pnt)
            np.testing.assert_allclose(res, pnt_vector)

        # Assert an error is raised if the point is not 3d
        with pytest.raises(ValueError):
            up([1, 2, 3, 4])

    def test_up_down_cga1(self, rng):  # noqa: F811
        """
        Test that we can map points up and down from cga1
        """
        from clifford.dg3c import up_cga1, down_cga1

        pnt_vector = rng.standard_normal(3)
        for i in range(10 if DISABLE_JIT else 100):
            pnt = up_cga1(pnt_vector)
            res = down_cga1(100*pnt)
            np.testing.assert_allclose(res, pnt_vector)

        # Assert an error is raised if the point is not 3d
        with pytest.raises(ValueError):
            up_cga1([1, 2, 3, 4])


@too_slow_without_jit
class TestGeometricPrimitives:
    @too_slow_without_jit
    def test_reciprocality(self):
        """
        Ensure that the cyclide ops and the reciprocal frame are
        actually reciprocal...
        """
        from clifford.dg3c import cyclide_ops, cyclide_ops_reciprocal

        for key, cyc_op in cyclide_ops.items():
            for key2, cyc_op_recip in cyclide_ops_reciprocal.items():
                assert cyc_op|cyc_op_recip == int(key == key2)

    def test_general_elipsoid(self):
        """
        Test the construction of a general elipsoid as per
        Appendix A.1
        """
        from clifford.dg3c import cyclide_ops, eo

        px = 0
        py = 1
        pz = 0
        rx = 3
        ry = 1
        rz = 2.5
        E = sum([
            (-2 * px / rx ** 2) * cyclide_ops['Tx'],
            (-2 * py / ry ** 2) * cyclide_ops['Ty'],
            (-2 * pz / rz ** 2) * cyclide_ops['Tz'],
            (1 / rx ** 2) * cyclide_ops['Tx2'],
            (1 / ry ** 2) * cyclide_ops['Ty2'],
            (1 / rz ** 2) * cyclide_ops['Tz2'],
            (px**2 / rx ** 2 + py**2 / ry ** 2 + pz**2 / rz ** 2 - 1) * cyclide_ops['T1']
        ])
        # The cyclides are an IPNS
        assert E|eo == 0

    def test_line(self, rng):  # noqa: F811
        from clifford.dg3c import up, up_cga1, up_cga2
        from clifford.dg3c import einf1, einf2, IC1, IC2

        # Make a dcga line
        pnt_vec_a = rng.standard_normal(3)
        pnt_vec_b = rng.standard_normal(3)
        Lcga1 = IC1*(up_cga1(pnt_vec_a) ^ up_cga1(pnt_vec_b) ^ einf1)
        Lcga2 = IC2*(up_cga2(pnt_vec_a) ^ up_cga2(pnt_vec_b) ^ einf2)
        Ldcga = Lcga1 ^ Lcga2

        # Assert that it is an IPNS
        assert Ldcga | up(pnt_vec_a) == 0
        assert Ldcga | up(pnt_vec_b) == 0
        assert Ldcga | up(0.5*pnt_vec_a + 0.5*pnt_vec_b) == 0

    @too_slow_without_jit
    def test_translation(self, rng):  # noqa: F811
        from clifford.dg3c import up, up_cga1, up_cga2
        from clifford.dg3c import cyclide_ops
        from clifford.dg3c import eo, e1, e2, e3, einf1, e6, e7, e8, einf2
        from clifford.dg3c import IC1, IC2

        # Make a dcga line
        pnt_vec = rng.standard_normal(3)
        direction_vec = rng.standard_normal(3)
        Lcga1 = IC1 * (up_cga1(pnt_vec) ^ up_cga1(pnt_vec + direction_vec) ^ einf1)
        Lcga2 = IC2 * (up_cga2(pnt_vec) ^ up_cga2(pnt_vec + direction_vec) ^ einf2)
        Ldcga = Lcga1 ^ Lcga2

        # Make a dcga translation rotor in direction of the line
        Tc1 = 1 - (direction_vec[0] * e1 + direction_vec[1] * e2 + direction_vec[2] * e3) * einf1
        Tc2 = 1 - (direction_vec[0] * e6 + direction_vec[1] * e7 + direction_vec[2] * e8) * einf2
        Tdcga = (Tc1*Tc2).normal()

        # Assert the rotor is normalised
        assert Tdcga*~Tdcga == 1

        # Apply the rotor to the line
        np.testing.assert_allclose((Tdcga*Ldcga*~Tdcga).value, Ldcga.value, rtol=1E-4, atol=1E-6)

        # Apply the rotor to a point on the line
        np.testing.assert_allclose(((Tdcga * up(pnt_vec) * ~Tdcga)|Ldcga).value, 0, rtol=1E-4, atol=1E-6)

        # Construct and ellipsoid at the origin
        px = 0
        py = 0
        pz = 0
        rx = 3
        ry = 1
        rz = 2.5
        E = sum([
            (-2 * px / rx ** 2) * cyclide_ops['Tx'],
            (-2 * py / ry ** 2) * cyclide_ops['Ty'],
            (-2 * pz / rz ** 2) * cyclide_ops['Tz'],
            (1 / rx ** 2) * cyclide_ops['Tx2'],
            (1 / ry ** 2) * cyclide_ops['Ty2'],
            (1 / rz ** 2) * cyclide_ops['Tz2'],
            (px ** 2 / rx ** 2 + py ** 2 / ry ** 2 + pz ** 2 / rz ** 2 - 1) * cyclide_ops['T1']
        ])
        # Before moving the elipsoid surface is not touching the origin
        assert E|eo != 0

        # Make a dcga translation rotor to move the ellipsoid
        Tc1 = 1 - 0.5 * rx * e1 * einf1
        Tc2 = 1 - 0.5 * rx * e6 * einf2
        Tdcga = (Tc1 * Tc2).normal()
        assert (Tdcga*E*~Tdcga) | eo == 0 * e1

        # Make a dcga translation rotor to move the ellipsoid
        Tc1 = 1 - 0.5 * ry * e2 * einf1
        Tc2 = 1 - 0.5 * ry * e7 * einf2
        Tdcga = (Tc1 * Tc2).normal()
        assert (Tdcga * E * ~Tdcga) | eo == 0 * e1

        # Make a dcga translation rotor to move the ellipsoid
        Tc1 = 1 - 0.5 * rz * e3 * einf1
        Tc2 = 1 - 0.5 * rz * e8 * einf2
        Tdcga = (Tc1 * Tc2).normal()
        assert (Tdcga * E * ~Tdcga) | eo == 0 * e1

    @too_slow_without_jit
    def test_line_rotation(self):
        from clifford.dg3c import up, up_cga1, up_cga2
        from clifford.dg3c import einf1, einf2
        from clifford.dg3c import e12, e67
        from clifford.dg3c import IC1, IC2

        theta = np.pi/2
        RC1 = np.e ** (-0.5*theta*e12)
        RC2 = np.e ** (-0.5*theta*e67)
        Rdcga = (RC1 * RC2).normal()
        assert Rdcga * ~Rdcga == 1

        # Construct a line
        pnt_vec = np.array([1, 0, 0])
        direction_vec = np.array([0, 0, 1])
        Lcga1 = IC1 * (up_cga1(pnt_vec) ^ up_cga1(pnt_vec + direction_vec) ^ einf1)
        Lcga2 = IC2 * (up_cga2(pnt_vec) ^ up_cga2(pnt_vec + direction_vec) ^ einf2)
        Ldcga = Lcga1 ^ Lcga2

        # Construct a second line
        pnt_vec_rotated = np.array([0, 1, 0])
        Lcga1_rotated = IC1 * (up_cga1(pnt_vec_rotated) ^ up_cga1(pnt_vec_rotated + direction_vec) ^ einf1)
        Lcga2_rotated = IC2 * (up_cga2(pnt_vec_rotated) ^ up_cga2(pnt_vec_rotated + direction_vec) ^ einf2)
        Ldcga_rotated = Lcga1_rotated ^ Lcga2_rotated

        # Assert the rotor rotates it
        assert (Rdcga * Ldcga * ~Rdcga)|up(pnt_vec_rotated) == 0
        np.testing.assert_allclose((Rdcga * Ldcga * ~Rdcga).value, Ldcga_rotated.value, rtol=1E-4, atol=1E-6)

    @too_slow_without_jit
    def test_quadric_rotation(self):
        # Construct and ellipsoid
        from clifford.dg3c import cyclide_ops
        from clifford.dg3c import eo, e2, e7, einf1, einf2
        from clifford.dg3c import e23, e78

        px = 0
        py = 2.5
        pz = 0
        rx = 3
        ry = 1
        rz = 2.5
        E = sum([
            (-2 * px / rx ** 2) * cyclide_ops['Tx'],
            (-2 * py / ry ** 2) * cyclide_ops['Ty'],
            (-2 * pz / rz ** 2) * cyclide_ops['Tz'],
            (1 / rx ** 2) * cyclide_ops['Tx2'],
            (1 / ry ** 2) * cyclide_ops['Ty2'],
            (1 / rz ** 2) * cyclide_ops['Tz2'],
            (px ** 2 / rx ** 2 + py ** 2 / ry ** 2 + pz ** 2 / rz ** 2 - 1) * cyclide_ops['T1']
        ])
        assert E | eo != 0

        # Make a dcga translation rotor to move the ellipsoid
        Tc1 = 1 - 0.5 * py * e2 * einf1
        Tc2 = 1 - 0.5 * py * e7 * einf2
        Tdcga = (Tc1 * Tc2).normal()

        # Construct a rotation rotor
        theta = np.pi / 2
        RC1 = np.e ** (-0.5 * theta * e23)
        RC2 = np.e ** (-0.5 * theta * e78)
        Rdcga = (RC1 * RC2).normal()

        Comborotor = (Tdcga*Rdcga*~Tdcga).normal()

        Erot = Comborotor*E*~Comborotor

        assert Erot|eo == 0

    @too_slow_without_jit
    def test_bivector_orthogonality(self):
        """
        Rotors in each algebra should be orthogonal
        """
        from clifford.dg3c import e2, e7, einf1, einf2
        from clifford.dg3c import e12, e67
        theta = np.pi / 2
        RC1 = np.e ** (-0.5 * theta * e12)
        RC2 = np.e ** (-0.5 * theta * e67)
        Rdcga = (RC1 * RC2).normal()
        Rexp = np.e**(-0.5*theta*(e12 + e67))
        np.testing.assert_allclose(Rexp.value, Rdcga.value,
                                   rtol=1E-4, atol=1E-6)

        mag = 5.0
        Tc1 = 1 - 0.5 * mag * e2 * einf1
        Tc2 = 1 - 0.5 * mag * e7 * einf2
        Tdcga = (Tc1 * Tc2).normal()
        Texp = np.e ** (-0.5 * mag * (e2 * einf1 + e7 * einf2))
        np.testing.assert_allclose(Texp.value, Tdcga.value,
                                   rtol=1E-4, atol=1E-6)
