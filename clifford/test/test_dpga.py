
import numpy as np
import pytest
from clifford.dpga import *


class TestBasicDPGA:
    def test_non_orthogonal_metric(self):
        for i, a in enumerate(wbasis):
            for j, b in enumerate(wbasis):
                res = a|b
                if i != j and i%4 == j%4:
                    assert res == (0.5) + 0*w1
                else:
                    assert res == 0*w1
                print(a|b, end=' ')
            print()

    def test_bivector_identities(self):
        for wi in wlist:
            for wj in wlist:
                assert wi^wj == -wj*wi

        for wis in wslist:
            for wjs in wslist:
                assert wis^wjs == -wjs*wis

        for w in wbasis:
            assert w**2 == 0*w1

        for wi, wis in zip(wlist, wslist):
            assert wi*wis == 1 - wis*wi

    def test_up_down(self):
        for i in range(1000):
            p = np.random.randn(3)
            dpga_pnt = up(p)
            pnt_down = down(np.random.rand()*dpga_pnt)
            np.testing.assert_allclose(pnt_down, p)

    def test_translate(self):
        for i in range(100):
            tvec = np.random.randn(3)
            wt = tvec[0]*w1 + tvec[1]*w2 + tvec[2]*w3
            biv = w0s*wt
            Rt = 1 - biv
            exp_result = np.e**(-biv)
            assert Rt == exp_result

            assert Rt * w0 * ~Rt == w0 + wt
            for wi in [w1, w2, w3]:
                assert Rt * wi * ~Rt == wi

            assert (Rt*~Rt) == 1 + 0*w1

            pnt_vec = np.random.randn(3)
            pnt = up(pnt_vec)
            res = Rt*pnt*~Rt
            desired_result = up(pnt_vec+tvec)

            assert up(pnt_vec) + wt == desired_result
            assert res == desired_result

            print()
            print(biv)
            print(tvec/4)
            print(biv*biv)
            print()

    def test_rotate(self):
        for i in range(100):
            mvec = np.random.randn(3)
            nvec = np.random.randn(3)
            m = mvec[0] * w1 + mvec[1] * w2 + mvec[2] * w3
            n = nvec[0] * w1 + nvec[1] * w2 + nvec[2] * w3
            ms = mvec[0] * w1s + mvec[1] * w2s + mvec[2] * w3s
            ns = nvec[0] * w1s + nvec[1] * w2s + nvec[2] * w3s
            biv = 2*((ms^n) - (ns^m))
            Rt = np.e**(-biv)

            # Rotor should be unit
            np.testing.assert_allclose((Rt*~Rt).value, (1 + 0*w1).value, atol=1E-4)

            # The origin should be unaffected by rotation
            np.testing.assert_allclose((Rt*w0*~Rt).value, w0.value, atol=1E-4)

            # Vectors orthogonal to the rotation should be unaffected by rotation
            vorthog = np.cross(mvec, nvec)
            uporthog = up(vorthog)
            np.testing.assert_allclose((Rt*uporthog*~Rt).value, uporthog.value, atol=1E-4)

            # Points should maintain their distance from the origin
            pnt_vec = np.random.randn(3)
            l = np.linalg.norm(pnt_vec)
            pnt = up(pnt_vec)
            lres = np.linalg.norm(down(Rt * pnt * ~Rt))
            np.testing.assert_allclose(l, lres, atol=1E-6)

            print()
            print(biv)
            print(vorthog)
            print()

    def test_line(self):
        for i in range(100):
            p1vec = np.random.randn(3)
            p2vec = np.random.randn(3)
            p1 = up(p1vec)
            p2 = up(p2vec)
            line_direc = p2vec - p1vec

            # Plucker line is the outer product of two points or the outer product
            # of a point and a free vector
            line = p1 ^ p2
            free_direc = line_direc[0]*w1 + line_direc[1]*w2 + line_direc[2]*w3
            line_alt = p1 ^ free_direc
            assert line_alt == line

            # The line should be the outer product null space
            lamb = np.random.randn()
            assert up(lamb*p1vec + (1 - lamb)*p2vec) ^ line == 0

            # Lines can be transformed with rotors
            tvec = p1vec
            wt = tvec[0] * w1 + tvec[1] * w2 + tvec[2] * w3
            Raxis = 1 - w0s * wt
            assert (~Raxis*line*Raxis) ^ w0 == 0

            # Lines are invariant to screw transformations about their axis
            axis = p1vec - p2vec
            rotation_biv = axis[0]*(e23 - e67) + axis[1]*(e57 - e13) + axis[2]*(e12 - e56)
            Rr = np.e**(-np.random.randn()*rotation_biv)
            Rt = 1 - w0s * np.random.randn()*(axis[0] * w1 + axis[1] * w2 + axis[2] * w3)
            np.testing.assert_allclose((Raxis*Rr*Rt*(~Raxis*line*Raxis)*~Rt*~Rr*~Raxis).value,
                                       line.value, rtol=1E-4, atol=1E-6)

            # A bivector line is invariant under its own exponential
            Rline = np.e ** line
            assert Rline*line*~Rline == line

            # The exponential of a line is a rotation about the line
            line_origin = (~Raxis*line*Raxis)
            RLineOrigin = np.e ** line_origin
            random_pnt = up(np.random.randn(3))
            np.testing.assert_allclose(np.linalg.norm(down(RLineOrigin*random_pnt*~RLineOrigin)),
                                       np.linalg.norm(down(random_pnt)), rtol=1E-3, atol=1E-4)
            np.testing.assert_allclose(down(RLineOrigin * (random_pnt + free_direc) * ~RLineOrigin),
                                       down(RLineOrigin * random_pnt * ~RLineOrigin) + line_direc, rtol=1E-3, atol=1E-4)
            np.testing.assert_allclose((Rline*~Rline).value, (1 + 0*w1).value, rtol=1E-4, atol=1E-4)

