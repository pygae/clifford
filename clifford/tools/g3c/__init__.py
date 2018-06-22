
import math
import numpy as np
from clifford.tools.g3 import quaternion_to_rotor, random_euc_mv
from clifford.g3c import *
import clifford as cf
import warnings

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


def generate_dilation_rotor(scale):
    """
    Generates a rotor that performs dilation about the origin
    """
    if abs(scale - 1.0) < 0.00001:
        u = np.zeros(32)
        u[0] = 1.0
        return cf.MultiVector(layout, u)
    gamma = math.log(scale)
    return math.cosh(gamma/2) + math.sinh(gamma/2)*(ninf^no)


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


def quaternion_and_vector_to_rotor(quaternion, vector):
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
    beta = math.sqrt(abs((T * T)[0]))
    F = T / beta
    P = 0.5 * (1 + F)
    P_twiddle = 0.5 * (1 - F)
    A = normalise_n_minus_1(-P_twiddle * (T | ninf))
    B = normalise_n_minus_1(P * (T | ninf))
    return A, B


def euc_dist(conf_mv_a, conf_mv_b):
    """ Returns the distance between two conformal points """
    dot_result = (conf_mv_a|conf_mv_b)[0]
    if dot_result < 0.0:
        return math.sqrt(-2.0*dot_result)
    else:
        return 0.0


def dorst_norm_val(sigma):
    """ Square Root of Rotors - Implements the norm of a rotor"""
    return math.sqrt(sigma[0] ** 2 - (sigma(4) ** 2)[0])


def check_sigma_for_positive_root(sigma):
    """ Square Root of Rotors - Checks for a positive root """
    return sigma[0] + dorst_norm_val(sigma) > 0


def check_sigma_for_negative_root(sigma):
    """ Square Root of Rotors - Checks for a negative root """
    return sigma[0] - dorst_norm_val(sigma) > 0


def check_infinite_roots(sigma):
    """ Square Root of Rotors - Checks for a infinite roots """
    return sigma[0] + dorst_norm_val(sigma) < 0.0000000000001


def positive_root(sigma):
    """
    Square Root of Rotors - Evaluates the positive root
    TODO: Dig out the full name of this paper and authors
    """
    norm_s = dorst_norm_val(sigma)
    return (sigma + norm_s) / (math.sqrt(2) * math.sqrt(sigma[0] + norm_s))


def negative_root(sigma):
    """ Square Root of Rotors - Evaluates the negative root """
    norm_s = dorst_norm_val(sigma)
    return (sigma - norm_s) / (math.sqrt(2) * math.sqrt(sigma[0] - norm_s))


def neg_twiddle_root(C):
    """
    Hadfield and Lasenby AGACSE2018
    Return a valid object from the addition result C
    """
    sigma = -(C * ~C)
    if check_sigma_for_positive_root(sigma):
        k = positive_root(sigma)
        C3 = ((1. / k) * C).normal()
        return [C3]
    elif check_sigma_for_negative_root(sigma):
        k = positive_root(sigma)
        C3 = ((1. / k) * C).normal()

        k2 = negative_root(sigma)
        C4 = ((1. / k2) * C).normal()
        return [C3, C4]
    elif check_infinite_roots(sigma):
        warnings.warn('Infinite roots detected: sigma = ' + str(sigma), RuntimeWarning)
        return [C.normal()]
    else:
        raise ValueError('No root exists')


def pos_twiddle_root(C):
    """
    Hadfield and Lasenby AGACSE2018
    Return a valid rotor from the (1 + C2C1) result C
    """
    sigma = (C * ~C)
    if check_sigma_for_positive_root(sigma):
        k = positive_root(sigma)
        C3 = ((1. / k) * C).normal()
        return [C3]
    elif check_sigma_for_negative_root(sigma):
        k = positive_root(sigma)
        C3 = ((1. / k) * C).normal()

        k2 = negative_root(sigma)
        C4 = ((1. / k2) * C).normal()
        return [C3, C4]
    elif check_infinite_roots(sigma):
        warnings.warn('Infinite roots detected: sigma = ' + str(sigma), RuntimeWarning)
        return [C.normal()]
    else:
        raise ValueError('No root exists')


def interp_objects_root(C1, C2, alpha):
    """
    Hadfield and Lasenby, Direct Linear Interpolation of Geometric Objects, AGACSE2018
    Directly linearly interpolates conformal objects
    """
    C = alpha * C1 + (1 - alpha) * C2
    return neg_twiddle_root(C)


def average_objects(obj_list):
    """
    Hadfield and Lasenby, Direct Linear Interpolation of Geometric Objects, AGACSE2018
    Directly averages conformal objects
    """
    C = sum(obj_list) / len(obj_list)
    return neg_twiddle_root(C)[0].normal()


def rotor_between_objects(C1, C2):
    """
    Hadfield and Lasenby AGACSE2018
    For any two conformal objects C1 and C2 this returns a rotor that takes C1 to C2
    """
    if float((C1 * C1)[0]) > 0:
        C = 1 + (C2 * C1)
        R = pos_twiddle_root(C)[0]
        return R
    else:
        return (1 - (C2 * C1)).normal()


def random_bivector():
    """
    Creates a random bivector on the form described by R. Wareham in
    TODO: Dig out the name of the interpolation paper

    $$ B =  ab + c*n_{\inf}$$ where $a, b, c \in \mathcal(R)^3$
    """
    a = random_euc_mv()
    c = random_euc_mv()
    return a * I3 + c * ninf


def random_point_pair():
    """
    Creates a random point pair bivector object
    """
    mv_a = random_euc_mv()
    mv_b = random_euc_mv()
    pp = (up(mv_a) ^ up(mv_b)).normal()
    return pp


def random_line():
    """
    Creates a random line
    """
    mv_a = random_euc_mv()
    mv_b = random_euc_mv()
    line_a = (up(mv_a) ^ up(mv_b) ^ ninf).normal()
    return line_a


def random_circle():
    """
    Creates a random circle
    """
    mv_a = random_euc_mv()
    mv_b = random_euc_mv()
    mv_c = random_euc_mv()
    line_a = (up(mv_a) ^ up(mv_b) ^ up(mv_c)).normal()
    return line_a


def random_sphere():
    """
    Creates a random sphere
    """
    mv_a = random_euc_mv()
    mv_b = random_euc_mv()
    mv_c = random_euc_mv()
    mv_d = random_euc_mv()
    sphere = (up(mv_a) ^ up(mv_b) ^ up(mv_c) ^ up(mv_d)).normal()
    return sphere


def random_plane():
    """
    Creates a random plane
    """
    line_a = (random_circle() ^ ninf).normal()
    return line_a
