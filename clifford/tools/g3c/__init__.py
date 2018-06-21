
import math
import numpy as np
from clifford.tools.g3 import quaternion_to_rotor, random_euc_mv
from clifford.g3c import *
import clifford as cf

# Allow sytactic alternatives to the standard included in the clifford package
ninf = einf
no = -eo

# Define some useful objects
E = ninf ^ (no)
I5 = e12345
I3 = e123


def random_conformal_point(l_max=10):
    """
    Creates a random conformal point
    """
    return up(random_euc_mv(l_max=l_max))


def generate_translation_rotor(euc_vector_a):
    """
    Generates a rotor that translates objects along the euclidean vector euc_vector_a
    """
    return (1 + ninf * euc_vector_a / 2)


def meet(A, B):
    """
    The meet algorithm as described in "A Covariant Approach to Geometry"
    """
    return I5*((I5*A) ^ (I5*B))


def intersect_line_and_plane_to_point(line, plane):
    """
    Returns the point at the intersection of a line and plane
    If there is no intersection it returns None
    """
    bivector = meet(line, plane).normal()
    if (bivector*bivector)[0] > 0:
        return bivector | no
    else:
        return None


def normalise_n_minus_1(mv):
    """
    Normalises a conformal point so that it has an inner product of -1 with einf
    """
    scale = (mv|ninf)[0]
    if scale != 0.0:
        return -mv/scale
    else:
        raise ZeroDivisionError('Multivector has 0 einf component')


def quaterion_and_vector_to_rotor(quaternion, vector):
    """
    Takes in a quaternion and a vector and returns a conformal rotor that
    implements the transformation
    """
    rotation = quaternion_to_rotor(quaternion)
    translation = generate_translation_rotor(
        vector[0] * e1 + vector[1] * e2 * vector[3] * e3)
    return translation * rotation


def get_center_from_sphere(sphere):
    """
    Returns the conformal point at the centre of a sphere by reflecting the
    point at infinity
    """
    center = sphere * ninf * sphere
    return center


def get_radius_from_sphere(sphere):
    """
    Returns the radius of a sphere
    """
    dual_sphere = sphere * I5
    dual_sphere = dual_sphere / (-dual_sphere | ninf)
    return math.sqrt(abs(dual_sphere * dual_sphere))


def point_pair_to_end_points(T):
    """
    Extracts the end points of a point pair bivector
    """
    beta = math.sqrt((T*T)[0])
    F = T/beta
    P = 0.5*(1+F)
    P_twiddle = 0.5*(1-F)
    A = -P_twiddle*(T|ninf)
    B = P*(T|ninf)
    return A, B


def euc_dist(conf_mv_a, conf_mv_b):
    """ Returns the distance between two conformal points """
    if (conf_mv_a|conf_mv_b)[0] < 0.0:
        return math.sqrt(-2.0*((conf_mv_a|conf_mv_b)[0]))
    else:
        return 0.0




