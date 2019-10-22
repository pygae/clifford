# https://clifford.readthedocs.io/en/latest/SpaceTimeAlgebra.html

from . import Cl, BladeMap

# Dirac Algebra  `D`
D, D_blades = Cl(1, 3, firstIdx=0, names='d')

# Pauli Algebra  `P`
P, P_blades = Cl(3, names='p')

# put elements of each in namespace
locals().update(D_blades)
locals().update(P_blades)

bm = BladeMap([(d01, p1),
               (d02, p2),
               (d03, p3),
               (d12, p12),
               (d23, p23),
               (d13, p13),
               (d0123, p123)])


def split(X):
    '''
    implements the spacetime split.
    '''
    return bm(X.odd*d0+X.even)
