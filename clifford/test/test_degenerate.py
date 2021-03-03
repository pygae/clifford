
from clifford import Cl
import math


class TestPGA:
    layout, _ = Cl(3, 0, 1, firstIdx=0)

    def test_right_complement(self):
        for b in self.layout.blades_list:
            assert b ^ b.right_complement() == self.layout.pseudoScalar

    def test_left_complement(self):
        for b in self.layout.blades_list:
            assert b.left_complement() ^ b == self.layout.pseudoScalar

    def test_vee(self):
        blades = self.layout.blades

        e0 = blades['e0']
        e1 = blades['e1']
        e2 = blades['e2']
        e3 = blades['e3']

        e01 = blades['e01']
        e02 = blades['e02']
        e03 = blades['e03']
        e12 = blades['e12']
        e13 = blades['e13']
        e23 = blades['e23']

        e123 = e1^e2^e3
        e032 = e0^e3^e2
        e013 = e0^e1^e3
        e021 = e0^e2^e1

        # PGA points are trivectors.
        # A point is just a homogeneous point, euclidean coordinates plus the origin
        def POINT(x, y, z):
            return e123 + x*e032 + y*e013 + z*e021

        P1 = POINT(0, 0, 1)
        P2 = POINT(3, 4, 0)
        L = P1 & P2
        assert L == -(4^e01) + (3^e02) - (1^e12) - (4^e13) + (3^e23)

    def test_no_crash(self):
        """ TODO: This doesn't actually do any asserts! """
        layout = self.layout
        blades = layout.blades
        print(blades)

        # A rotor (Euclidean line) and translator (Ideal line)
        def rotor(angle, line):
            return math.cos(angle/2.0) + math.sin(angle/2.0)*line.normal()

        def translator(dist, line):
            return 1.0 + dist/2.0*line

        # PGA is plane based. Vectors are planes. (think linear functionals)
        E0 = blades['e0']          # ideal plane
        E1 = blades['e1']           # x=0 plane
        E2 = blades['e2']           # y=0 plane
        E3 = blades['e3']           # z=0 plane

        # A plane is defined using its homogenous equation ax + by + cz + d = 0
        def PLANE(a, b, c, d):
            return a*E1 + b*E2 + c*E3 + d*E0

        # PGA points are trivectors.
        E123 = E1^E2^E3
        E032 = E0^E3^E2
        E013 = E0^E1^E3
        E021 = E0^E2^E1

        # A point is just a homogeneous point, euclidean coordinates plus the origin
        def POINT(x, y, z):
            return E123 + x*E032 + y*E013 + z*E021

        # for our toy problem (generate points on the surface of a torus)
        # we start with a function that generates motors.
        # circle(t) with t going from 0 to 1.
        def CIRCLE(t, radius, line):
            return rotor(t*math.pi*2.0, line)*translator(radius, E1*E0)

        # a torus is now the product of two circles.
        def TORUS(s, t, r1, l1, r2, l2):
            return CIRCLE(s, r2, l2)*CIRCLE(t, r1, l1)

        # sample the torus points by sandwich with the origin
        def POINT_ON_TORUS(s, t):
            to = TORUS(s, t, 0.25, E1*E2, 0.6, E1*E3)
            return to * E123 * ~to

        # Elements of the even subalgebra (scalar + bivector + pss) of unit length are motors
        ROT = rotor(math.pi/2.0, E1*E2)

        # The outer product ^ is the MEET. Here we intersect the yz (x=0) and xz (y=0) planes.
        AXZ = E1 ^ E2                # x=0, y=0 -> z-axis line

        # line and plane meet in point. We intersect the line along the z-axis (x=0, y=0) with the xy (z=0) plane.
        ORIG = AXZ ^ E3              # x=0, y=0, z=0 -> origin

        # We can also easily create points and join them into a line using the regressive (vee, &) product.
        PX = POINT(1, 0, 0)
        LINE = (ORIG.dual()^PX.dual()).dual()             # & = regressive product, JOIN, here, x-axis line.

        # Lets also create the plane with equation 2x + z - 3 = 0
        P = PLANE(2, 0, 1, -3)

        # rotations work on all elements ..
        ROTATED_LINE = ROT * LINE * ~ROT
        ROTATED_POINT = ROT * PX * ~ROT
        ROTATED_PLANE = ROT * P * ~ROT

        # See the 3D PGA Cheat sheet for a huge collection of useful formulas
        POINT_ON_PLANE = (P | PX) * P

        # output some numbers.
        print("a point       :", str(PX))
        print("a line        :", str(LINE))
        print("a plane       :", str(P))
        print("a rotor       :", str(ROT))
        print("rotated line  :", str(ROTATED_LINE))
        print("rotated point :", str(ROTATED_POINT))
        print("rotated plane :", str(ROTATED_PLANE))
        print("point on plane:", str(POINT_ON_PLANE.normal()))
        print("point on torus:", str(POINT_ON_TORUS(0.0, 0.0)))
