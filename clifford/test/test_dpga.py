
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

            print()
            print(biv)
            print(vorthog)
            print()
