"""
The Cl(8,2) Double Conformal Geometric Algebra

Easter, R.B., Hitzer, E. Double Conformal Geometric Algebra.
Adv. Appl. Clifford Algebras 27, 2175–2199 (2017).
https://doi.org/10.1007/s00006-017-0784-0
"""

from . import Layout

# The metric of the algebra is two CGAs glued together
layout = Layout([1]*4+[-1] + [1]*4+[-1])
blades = layout.bases()

locals().update(blades)


# for shorter reprs
layout.__name__ = 'layout'
layout.__module__ = __name__


# The two pseudo-scalars, infinities and origins
IC1 = e12345
IC2 = e678910
einf1 = e4 + e5
eo1 = 0.5*(e5 - e4)
einf2 = e9 + e10
eo2 = 0.5*(e10 - e9)

# The joint infinity and origin
eo = eo1^eo2
einf = einf1^einf2


def up_cga1(pnt_vector):
    """
    Take a vector and embed it as a point in the first
    copy of cga
    """
    x, y, z = pnt_vector
    euc_point = x*e1 + y*e2 + z*e3
    return euc_point + 0.5*(euc_point|euc_point)*einf1 + eo1


def down_cga1(point_cga1):
    """
    Take a point in CGA
    """
    return (point_cga1/-(point_cga1|einf1)[()]).value[1:4]


def up_cga2(pnt_vector):
    """
    Take a vector and embed it as a point in the second
    copy of cga
    """
    x, y, z = pnt_vector
    euc_point = x*e6 + y*e7 + z*e8
    return euc_point + 0.5*euc_point**2*einf2 + eo2


def up(pnt_vector):
    """
    Take a vector and embed it as a dcga point
    """
    return up_cga1(pnt_vector)^up_cga2(pnt_vector)


def down(dcga_point):
    """
    Take a dcga_point and return the 3d vector it represents
    """
    cga_pnt = ((dcga_point|einf2)|IC1)*IC1
    return down_cga1(cga_pnt)


"""
These cyclide_ops are the elements that make up a general Darboux cyclide
See Table 1 and Table 2 from Easter, Hitzer, Double Conformal Geometric Algebra (2017)
"""
cyclide_ops = {
    "Tx": 0.5 * (e1 * einf2 + einf1 * e6),
    "Ty":  0.5 * (e2 * einf2 + einf1 * e7),
    "Tz":  0.5 * (e3 * einf2 + einf1 * e8),

    "Tx2":  e6 * e1,
    "Ty2":  e7 * e2,
    "Tz2":  e8 * e3,

    "Txy":  0.5 * (e7 * e1 + e6 * e2),
    "Tyz":  0.5 * (e7 * e3 + e8 * e2),
    "Tzx":  0.5 * (e8 * e1 + e6 * e3),

    "Txt2":  e1 * eo2 + eo1 * e6,
    "Tyt2":  e2 * eo2 + eo1 * e7,
    "Tzt2":  e3 * eo2 + eo1 * e8,

    "T1":  -einf,
    "Tt2":  eo2*einf1 + einf2*eo1,
    "Tt4":  -4*eo
}

cyclide_ops_reciprocal = {
    "Tx":  cyclide_ops['Txt2'],
    "Ty":  cyclide_ops['Tyt2'],
    "Tz":  cyclide_ops['Tzt2'],

    "Tx2":  -cyclide_ops['Tx2'],
    "Ty2":  -cyclide_ops['Ty2'],
    "Tz2":  -cyclide_ops['Tz2'],

    "Txy":  -2*cyclide_ops['Txy'],
    "Tyz":  -2*cyclide_ops['Tyz'],
    "Tzx":  -2*cyclide_ops['Tzx'],

    "Txt2":  cyclide_ops['Tx'],
    "Tyt2":  cyclide_ops['Ty'],
    "Tzt2":  cyclide_ops['Tz'],

    "T1":  -cyclide_ops['Tt4']/4,
    "Tt2":  -cyclide_ops['Tt2']/2,
    "Tt4":  -cyclide_ops['T1']/4
}

