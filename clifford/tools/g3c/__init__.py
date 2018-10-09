"""
Tools for 3DCGA (g3c)

3DCGA Tools
==========================================================

Generation Methods
--------------------

.. autosummary::
    :toctree: generated/

    random_bivector
    standard_point_pair_at_origin
    random_point_pair_at_origin
    random_point_pair
    standard_line_at_origin
    random_line_at_origin
    random_line
    random_circle_at_origin
    random_circle
    random_sphere_at_origin
    random_sphere
    random_plane_at_origin
    random_plane

    generate_n_clusters
    generate_random_object_cluster
    random_translation_rotor
    random_rotation_translation_rotor
    random_conformal_point
    generate_dilation_rotor
    generate_translation_rotor


Geometry Methods
--------------------

.. autosummary::
    :toctree: generated/

    intersect_line_and_plane_to_point
    quaternion_and_vector_to_rotor
    get_center_from_sphere
    get_radius_from_sphere
    point_pair_to_end_points
    get_circle_in_euc
    circle_to_sphere
    line_to_point_and_direction
    get_plane_origin_distance
    get_plane_normal
    get_nearest_plane_point
    val_convert_2D_polar_line_to_conformal_line
    convert_2D_polar_line_to_conformal_line
    val_convert_2D_point_to_conformal
    convert_2D_point_to_conformal
    val_distance_point_to_line
    distance_polar_line_to_euc_point_2d


Misc
--------------------

.. autosummary::
    :toctree: generated/
    
    
    meet_val
    meet
    normalise_n_minus_1
    val_apply_rotor
    apply_rotor
    val_apply_rotor_inv
    apply_rotor_inv
    euc_dist
    mult_with_ninf
    val_norm
    norm
    val_normalised
    normalised
    val_up
    fast_up
    val_normalInv
    val_homo
    val_down
    fast_down
    dual_func
    fast_dual
    disturb_object
    project_val


Root Finding
--------------------

.. autosummary::
    :toctree: generated/
    
    dorst_norm_val
    check_sigma_for_positive_root_val
    check_sigma_for_positive_root
    check_sigma_for_negative_root_val
    check_sigma_for_negative_root
    check_infinite_roots_val
    check_infinite_roots
    positive_root_val
    negative_root_val
    positive_root
    negative_root
    general_root_val
    general_root
    val_annihilate_k
    annihilate_k
    pos_twiddle_root_val
    neg_twiddle_root_val
    pos_twiddle_root
    neg_twiddle_root
    square_roots_of_rotor
    interp_objects_root
    average_objects
    rotor_between_objects
    val_rotor_between_objects
    val_rotor_between_lines
    rotor_between_lines
    rotor_between_planes
    val_rotor_rotor_between_planes

"""

import math
import numba
import numpy as np
from clifford.tools.g3 import quaternion_to_rotor, random_euc_mv, \
    random_rotation_rotor, generate_rotation_rotor, val_random_euc_mv
from clifford.g3c import *
import clifford as cf
import warnings

# Allow syntactic alternatives to the standard included in the clifford package
ninf = einf
no = -eo

# Define some useful objects
E = ninf ^ (no)
I5 = e12345
I3 = e123
E0 = ninf ^ -no
niono = ninf ^ no
E0_val = E0.value
I5_val = I5.value
ninf_val = ninf.value
no_val = no.value
I3_val = I3.value

unit_scalar_mv = 1.0 + 0.0*e1
unit_scalar_mv_val = unit_scalar_mv.value

adjoint_func = layout.adjoint_func
gmt_func = layout.gmt_func
omt_func = layout.omt_func
imt_func = layout.imt_func

epsilon = 10*10**(-6)


def interpret_multivector_as_object(mv):
    """
    Takes an input multivector and returns what kind of object it is
    -1 -> not a blade
    0 -> a 1 vector but not a point
    1 -> a euclidean point
    2 -> a conformal point
    3 -> a point pair
    4 -> a circle
    5 -> a line
    6 -> a sphere
    7 -> a plane
    """
    grades_present = mv.grades()
    if len(grades_present) != 1:  # Definitely not a blade
        return -1
    grade = grades_present[0]
    if grade == 1:
        # This can now either be a euc mv or a conformal point or something else
        if mv(e123) == mv:
            return 1  # Euclidean point
        elif mv**2 == 0*e1:
            return 2  # Conformal point
        else:
            return 0  # Unknown mv
    elif mv.isBlade():
        if grade == 2:
            return 3  # Point pair
        elif grade == 3:  # Line or circle
            if abs(mv[e123]) > epsilon:
                return 4  # Circle
            else:
                return 5  # Line
        elif grade == 4:  # Sphere or plane
            if abs(((mv*I5)|ninf)[0]) > epsilon:
                return 7  # Plane
            else:
                return 6  # Sphere
    else:
        return -1

def get_circle_in_euc(circle):
    """ Extracts all the normal stuff for a circle """
    Ic = (circle^ninf).normal()
    GAnormal = get_plane_normal(Ic)
    inPlaneDual = circle*Ic
    mag = float((inPlaneDual|ninf)[0])
    inPlaneDual = -inPlaneDual/mag
    radius_squared = (inPlaneDual*inPlaneDual)[0]
    radius = math.sqrt(abs(radius_squared))
    if radius_squared < 0:
        # We have an imaginary circle, return it as a negative radius as our signal
        radius = -radius
    GAcentre = down(inPlaneDual*(1+0.5*inPlaneDual*ninf))
    return [GAcentre,GAnormal,radius]


def circle_to_sphere(C):
    """
    returns the sphere for which the input circle is the perimeter
    """
    Ic = (C ^ einf).normal()
    sphere = C * Ic * I5
    return sphere


def line_to_point_and_direction(line):
    """ 
    converts a line to the conformal nearest point to the origin and a 
    euc direction vector in direction of the line 
    """
    L_star = line*I5
    T = L_star|no
    mhat = -(L_star - T*ninf)*I3
    p = (T^mhat)*I3
    return [p,mhat]


def get_plane_origin_distance(plane):
    """ Get the distance between a given plane and the origin """
    return float(((plane*I5)|no)[0])


def get_plane_normal(plane):
    """ Get the normal to the plane """
    return (plane*I5 - get_plane_origin_distance(plane)*ninf)


def get_nearest_plane_point(plane):
    """ Get the nearest point to the origin on the plane """
    return get_plane_normal(plane)*get_plane_origin_distance(plane)


def disturb_object(mv_object, maximum_translation=0.01, maximum_angle=0.01):
    """ Disturbs an object by a random rotor """
    r = random_rotation_translation_rotor(maximum_translation=maximum_translation, maximum_angle=maximum_angle)
    return (r*mv_object*~r).normal()


def generate_n_clusters( object_generator, n_clusters, n_objects_per_cluster ):
    """ Creates n_clusters of random objects """
    object_clusters = []
    for i in range(n_clusters):
        cluster_objects = generate_random_object_cluster(n_objects_per_cluster, object_generator,
                                                         max_cluster_trans=0.5, max_cluster_rot=np.pi / 16)
        object_clusters.append(cluster_objects)
    all_objects = [item for sublist in object_clusters for item in sublist]
    return all_objects, object_clusters


def generate_random_object_cluster(n_objects, object_generator, max_cluster_trans=1.0, max_cluster_rot=np.pi/8):
    """ Creates a cluster of random objects """
    ref_obj = object_generator()
    cluster_objects = []
    for i in range(n_objects):
        r = random_rotation_translation_rotor(maximum_translation=max_cluster_trans, maximum_angle=max_cluster_rot)
        new_obj = apply_rotor(ref_obj, r)
        cluster_objects.append(new_obj)
    return cluster_objects


def random_translation_rotor(maximum_translation=10.0):
    """ generate a random translation rotor """
    return generate_translation_rotor(random_euc_mv(maximum_translation))


def random_rotation_translation_rotor(maximum_translation=10.0, maximum_angle=np.pi):
    """ generate a random combined rotation and translation rotor """
    return (random_translation_rotor(maximum_translation)*random_rotation_rotor(maximum_angle)).normal()


@numba.njit
def project_val(val, grade):
    """ fast grade projection """
    output = np.zeros(32)
    if grade == 0:
        output[0] = val[0]
    elif grade == 1:
        output[1:6] = val[1:6]
    elif grade == 2:
        output[6:16] = val[6:16]
    elif grade == 3:
        output[16:26] = val[16:26]
    elif grade == 4:
        output[26:31] = val[26:31]
    elif grade == 5:
        output[31] = val[31]
    return output


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


@numba.njit
def meet_val(a_val, b_val):
    """
    The meet algorithm as described in "A Covariant Approach to Geometry"
    I5*((I5*A) ^ (I5*B))
    """
    return dual_func(omt_func(dual_func(a_val), dual_func(b_val)))


def meet(A, B):
    """
    The meet algorithm as described in "A Covariant Approach to Geometry"
    I5*((I5*A) ^ (I5*B))
    """
    return cf.MultiVector(layout, meet_val(A.value, B.value))


def intersect_line_and_plane_to_point(line, plane):
    """
    Returns the point at the intersection of a line and plane
    If there is no intersection it returns None
    """
    biv = meet(line, plane)
    abs_biv = abs(biv)
    if abs_biv > 0.000001:
        if (biv|niono)[0] > 0:
            return ((biv ^ no) | niono) / abs(biv)
        else:
            return -((biv ^ no) | niono) / abs(biv)
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
    dual_sphere = dual_sphere / (-dual_sphere | ninf)[0]
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


@numba.jit
def dorst_norm_val(sigma_val):
    """ Square Root of Rotors - Implements the norm of a rotor"""
    sigma_4 = project_val(sigma_val, 4)
    sqrd_ans = sigma_val[0] ** 2 - gmt_func(sigma_4,sigma_4)[0]
    return math.sqrt(sqrd_ans)


@numba.njit
def check_sigma_for_positive_root_val(sigma_val):
    """ Square Root of Rotors - Checks for a positive root """
    return (sigma_val[0] + dorst_norm_val(sigma_val)) > 0


def check_sigma_for_positive_root(sigma):
    """ Square Root of Rotors - Checks for a positive root """
    return check_sigma_for_positive_root_val(sigma.value)


@numba.njit
def check_sigma_for_negative_root_val(sigma_value):
    """ Square Root of Rotors - Checks for a negative root """
    return (sigma_value[0] - dorst_norm_val(sigma_value)) > 0


def check_sigma_for_negative_root(sigma):
    """ Square Root of Rotors - Checks for a negative root """
    return check_sigma_for_negative_root_val(sigma.value)


@numba.njit
def check_infinite_roots_val(sigma_value):
    """ Square Root of Rotors - Checks for a infinite roots """
    return (sigma_value[0] + dorst_norm_val(sigma_value)) < 0.0000000001


def check_infinite_roots(sigma):
    """ Square Root of Rotors - Checks for a infinite roots """
    return check_infinite_roots_val(sigma.value)


@numba.njit
def positive_root_val(sigma_val):
    """
    Square Root of Rotors - Evaluates the positive root
    Square Root of Rotors - Evaluates the positive root
    """
    norm_s = dorst_norm_val(sigma_val)
    denominator = (math.sqrt(2) * math.sqrt(sigma_val[0] + norm_s))
    result = sigma_val/denominator
    result[0] += norm_s/denominator
    return result


@numba.njit
def negative_root_val(sigma_val):
    """
    Square Root of Rotors - Evaluates the positive root
    """
    norm_s = dorst_norm_val(sigma_val)
    denominator = (math.sqrt(2) * math.sqrt(sigma_val[0] - norm_s))
    result = sigma_val/denominator
    result[0] -= norm_s/denominator
    return result


def positive_root(sigma):
    """
    Square Root of Rotors - Evaluates the positive root
    """
    res_val = positive_root_val(sigma.value)
    return cf.MultiVector(layout, res_val)


def negative_root(sigma):
    """ Square Root of Rotors - Evaluates the negative root """
    res_val = negative_root_val(sigma.value)
    return cf.MultiVector(layout, res_val)


@numba.njit
def general_root_val(sigma_value):
    """
    Square Root and Logarithm of Rotors
    in 3D Conformal Geometric Algebra
    Using Polar Decomposition
    Leo Dorst and Robert Valkenburg
    """
    if check_sigma_for_positive_root_val(sigma_value):
        k = positive_root_val(sigma_value)
        output = np.zeros((2, sigma_value.shape[0]))
        output[0, :] = k.copy()
        return output
    elif check_sigma_for_negative_root_val(sigma_value):
        k = positive_root_val(sigma_value)
        k2 = negative_root_val(sigma_value)
        output = np.zeros((2, sigma_value.shape[0]))
        output[0, :] = k.copy()
        output[1, :] = k2.copy()
        return output
    elif check_infinite_roots_val(sigma_value):
        output = np.zeros((2, sigma_value.shape[0]))
        output[:, 0] = 1.0
        return output
    else:
        raise ValueError('No root exists')


def general_root(sigma):
    """ The general case of the root of a grade 0,4 multivector """
    output = general_root_val(sigma.value)
    return [cf.MultiVector(layout, output[0, :].copy()), cf.MultiVector(layout, output[1, :].copy())]


@numba.njit
def val_annihilate_k(K_val, C_val):
    """ Removes K from C = KX via (K[0] - K[4])*C """
    k_4 = -project_val(K_val, 4)
    k_4[0] += K_val[0]
    return val_normalised(gmt_func(k_4, C_val))


def annihilate_k(K, C):
    """ Removes K from C = KX via (K[0] - K[4])*C """
    return cf.MultiVector(layout, val_annihilate_k(K.value, C.value))


@numba.njit
def pos_twiddle_root_val(C_value):
    """
    Square Root and Logarithm of Rotors
    in 3D Conformal Geometric Algebra
    Using Polar Decomposition
    Leo Dorst and Robert Valkenburg
    """
    sigma_val = gmt_func(C_value, adjoint_func(C_value))
    k_value = general_root_val(sigma_val)
    output = np.zeros((2,32))
    output[0, :] = val_annihilate_k(k_value[0, :], C_value)
    output[1, :] = val_annihilate_k(k_value[1, :], C_value)
    return output


@numba.njit
def neg_twiddle_root_val(C_value):
    """
    Square Root and Logarithm of Rotors
    in 3D Conformal Geometric Algebra
    Using Polar Decomposition
    Leo Dorst and Robert Valkenburg
    """
    sigma_val = -gmt_func(C_value, adjoint_func(C_value))
    k_value = general_root_val(sigma_val)
    output = np.zeros((2, 32))
    output[0, :] = val_annihilate_k(k_value[0, :], C_value)
    output[1, :] = val_annihilate_k(k_value[1, :], C_value)
    return output


def pos_twiddle_root(C):
    """
    Square Root and Logarithm of Rotors
    in 3D Conformal Geometric Algebra
    Using Polar Decomposition
    Leo Dorst and Robert Valkenburg
    """
    output = pos_twiddle_root_val(C.value)
    return [cf.MultiVector(layout, output[0, :]), cf.MultiVector(layout, output[1, :])]


def neg_twiddle_root(C):
    """
    Square Root and Logarithm of Rotors
    in 3D Conformal Geometric Algebra
    Using Polar Decomposition
    Leo Dorst and Robert Valkenburg
    """
    output = neg_twiddle_root_val(C.value)
    return [cf.MultiVector(layout, output[0, :]), cf.MultiVector(layout, output[1, :])]


def square_roots_of_rotor(R):
    """
    Square Root and Logarithm of Rotors
    in 3D Conformal Geometric Algebra
    Using Polar Decomposition
    Leo Dorst and Robert Valkenburg
    """
    return pos_twiddle_root(1 + R)


def n_th_rotor_root(R, n):
    """
    Takes the n_th root of rotor R
    n must be a power of 2
    """
    if not (((n & (n - 1)) == 0) and n != 0):
        raise ValueError('n is not a power of 2')
    if n == 1:
        return R
    else:
        return n_th_rotor_root(square_roots_of_rotor(R)[0], int(n/2))


def interp_objects_root(C1, C2, alpha):
    """
    Hadfield and Lasenby, Direct Linear Interpolation of Geometric Objects, AGACSE2018
    Directly linearly interpolates conformal objects
    Return a valid object from the addition result C
    """
    C = (1 - alpha) * C1 + alpha*C2
    C3 = (neg_twiddle_root(C)[0]).normal()
    if cf.grade_obj(C1, 0.00001) != cf.grade_obj(C3, 0.00001):
        raise ValueError('Created object is not same grade')
    return C3


from scipy.interpolate import interp1d
def general_object_interpolation(object_alpha_array, object_list, new_alpha_array, kind='linear'):
    """
    Hadfield and Lasenby, Direct Linear Interpolation of Geometric Objects, AGACSE2018
    This is a general interpolation through the
    """
    obj_array = np.transpose(ConformalMVArray(object_list).value)
    f = interp1d(object_alpha_array, obj_array, kind=kind)
    new_value_array = np.transpose(f(new_alpha_array))
    new_conf_array = ConformalMVArray.from_value_array(new_value_array)
    return [(neg_twiddle_root(C)[0]).normal() for C in new_conf_array]


def average_objects(obj_list, weights=[]):
    """
    Hadfield and Lasenby, Direct Linear Interpolation of Geometric Objects, AGACSE2018
    Directly averages conformal objects
    Return a valid object from the addition result C
    """
    if len(weights) == len(obj_list):
        C = sum([o * w for o, w in zip(obj_list, weights)])
    else:
        C = sum(obj_list) / len(obj_list)
    C3 = (neg_twiddle_root(C)[0]).normal()
    if cf.grade_obj(obj_list[0], 0.00001) != cf.grade_obj(C3, 0.00001):
        raise ValueError('Created object is not same grade')
    return C3


def rotor_between_objects(X1, X2):
    """
    Lasenby and Hadfield AGACSE2018
    For any two conformal objects X1 and X2 this returns a rotor that takes X1 to X2
    Return a valid object from the addition result 1 + gamma*X2X1
    """
    return cf.MultiVector(layout,
                          val_rotor_between_objects_root(val_normalised(X1.value),
                                                    val_normalised(X2.value)))


def calculate_S_over_mu(X1, X2):
    """
    Lasenby and Hadfield AGACSE2018
    For any two conformal objects X1 and X2 this returns a factor that corrects
    the X1 + X2 back to a blade
    """
    gamma1 = (X1 * X1)[0]
    gamma2 = (X2 * X2)[0]

    M12 = X1 * X2 + X2 * X1
    K = 2 + gamma1 * M12
    mu = (K[0]**2 - K(4)**2)[0]

    if sum(np.abs(M12(4).value)) > 0.0000001:
        lamb = (-(K(4) * K(4)))[0]
        mu = K[0] ** 2 + lamb
        root_mu = np.sqrt(mu)
        if abs(lamb) < 0.0000001:
            beta = 1.0 / (2 * np.sqrt(K[0]))
        else:
            beta_sqrd = 1 / (2 * (root_mu + K[0]))
            beta = np.sqrt(beta_sqrd)
        S = -gamma1/(2*beta) + beta*M12(4)
        return S/np.sqrt(mu)
    else:
        S = np.sqrt(abs(K[0]))
    return S/np.sqrt(mu)


@numba.njit
def val_rotor_between_objects_root(X1, X2):
    """
    Lasenby and Hadfield AGACSE2018
    For any two conformal objects X1 and X2 this returns a rotor that takes X1 to X2
    Uses the square root of rotors for efficiency and numerical stability
    """
    gamma = gmt_func(X1, X1)[0]
    C_val = gamma*gmt_func(X2, X1)
    C_val[0] += 1
    return val_normalised(pos_twiddle_root_val(C_val)[0, :])


@numba.njit
def val_rotor_between_objects_explicit(X1, X2):
    """
    Lasenby and Hadfield AGACSE2018
    For any two conformal objects X1 and X2 this returns a rotor that takes X1 to X2

    Implements an optimised version of:

    gamma1 = (X1 * X1)[0]
    gamma2 = (X2 * X2)[0]

    M12 = X1*X2 + X2*X1
    K = 2 + gamma1*M12

    if np.sum(np.abs(K.value)) < 0.0000001:
        return 1 + 0*e1

    if sum(np.abs(M12(4).value)) > 0.0000001:
        lamb = (-(K(4) * K(4)))[0]
        mu = K[0]**2 + lamb
        root_mu = np.sqrt(mu)
        if abs(lamb) < 0.0000001:
            beta = 1.0/(2*np.sqrt(K[0]))
        else:
            beta_sqrd = 1/(2*(root_mu + K[0]))
            beta = np.sqrt(beta_sqrd)
        R = ( beta*K(4) - (1/(2*beta)) )*(1 + gamma1*X2*X1)/(root_mu)
        return R
    else:
        return (1 + gamma1*X2*X1)/(np.sqrt(abs(K[0])))

    """
    gamma1 = gmt_func(X1, X1)[0]
    gamma2 = gmt_func(X2, X2)[0]

    # if abs(abs(gamma1) - 1.0) > 0.00001:
    #     raise ValueError('X1 must be normalised to give abs(X1*X1) == 1')
    # elif abs(abs(gamma2) - 1.0) > 0.00001:
    #     raise ValueError('X1 must be normalised to give abs(X1*X1) == 1')
    # elif abs(gamma1 - gamma2) > 0.00001:
    #     raise ValueError('X1 and X2 must square to the same value')

    M12_val = gmt_func(X1, X2) + gmt_func(X2, X1)
    K_val = gamma1 * M12_val
    K_val[0] = K_val[0] + 2

    if np.sum(np.abs(K_val)) < 0.0000001:
        return unit_scalar_mv_val

    if np.sum(np.abs(project_val(M12_val, 4))) > 0.00001:
        K_val_4 = project_val(K_val, 4)
        lamb = -gmt_func(K_val_4, K_val_4)[0]
        mu = K_val[0] ** 2 + lamb

        root_mu = np.sqrt(mu)
        beta_sqrd = 1 / (2 *root_mu + 2 *K_val[0])
        beta = np.sqrt(beta_sqrd)

        temp_1 = beta * K_val_4
        temp_1[0] = temp_1[0] - (1 / (2 * beta))

        temp_2 = gamma1*gmt_func(X2,X1)
        temp_2[0] = temp_2[0] + 1.0

        R_val = gmt_func(temp_1, temp_2) / root_mu
        return R_val
    else:
        temp_1 = gamma1 * gmt_func(X2, X1)
        temp_1[0] += 1.0
        return temp_1/(np.sqrt(abs(K_val[0])))


sparse_line_gmt = layout.gmt_func_generator(grades_a=[3], grades_b=[3])


@numba.njit
def val_norm(mv_val):
    """ Returns sqrt(abs(~A*A)) """
    return np.sqrt(np.abs(gmt_func(adjoint_func(mv_val), mv_val)[0]))


def norm(mv):
    """ Returns sqrt(abs(~A*A)) """
    return val_norm(mv.value)


@numba.njit
def val_normalised(mv_val):
    """ Returns A/sqrt(abs(~A*A)) """
    return mv_val/val_norm(mv_val)


def normalised(mv):
    """ fast version of the normal() function """
    return cf.MultiVector(layout, val_normalised(mv.value))


@numba.njit
def val_rotor_between_lines(L1_val, L2_val):
    """ Implements a very optimised rotor line to line extraction """
    L21_val = sparse_line_gmt(L2_val, L1_val)
    L12_val = sparse_line_gmt(L1_val, L2_val)
    K_val = L21_val + L12_val
    K_val[0] += 2.0
    beta_val = project_val(K_val, 4)
    alpha = 2 * K_val[0]

    denominator = np.sqrt(alpha / 2)
    numerator_val = -beta_val/alpha
    numerator_val[0] += 1.0
    normalisation_val = numerator_val/denominator

    output_val = L21_val
    output_val[0] += 1
    return gmt_func(normalisation_val, output_val)


def rotor_between_lines(L1, L2):
    """ return the rotor between two lines """
    return cf.MultiVector(layout, val_rotor_between_lines(L1.value, L2.value))


def rotor_between_planes(P1, P2):
    """ return the rotor between two planes """
    return cf.MultiVector(layout, val_rotor_rotor_between_planes(P1.value, P2.value))


@numba.njit
def val_rotor_rotor_between_planes(P1_val, P2_val):
    """ return the rotor between two planes """
    P21_val = -gmt_func(P2_val, P1_val)
    P21_val[0] += 1
    return val_normalised(P21_val)


def random_bivector():
    """
    Creates a random bivector on the form described by R. Wareham in
    Mesh Vertex Pose and Position Interpolation using Geometric Algebra.
    $$ B =  ab + c*n_{\inf}$$ where $a, b, c \in \mathcal(R)^3$
    """
    a = random_euc_mv()
    c = random_euc_mv()
    return a * I3 + c * ninf


def standard_point_pair_at_origin():
    """ Creates a standard point pair at the origin """
    return (up(-0.5*e1)^up(0.5*e1)).normal()


def random_point_pair_at_origin():
    """
    Creates a random point pair bivector object at the origin
    """
    mv_a = random_euc_mv()
    plane_a = (mv_a*I3).normal()

    mv_b = plane_a*mv_a*plane_a
    pp = (up(mv_a) ^ up(mv_b)).normal()
    return pp


def random_point_pair():
    """
    Creates a random point pair bivector object
    """
    mv_a = random_euc_mv()
    mv_b = random_euc_mv()
    pp = (up(mv_a) ^ up(mv_b)).normal()
    return pp


def standard_line_at_origin():
    return (standard_point_pair_at_origin()^einf).normal()


def random_line_at_origin():
    """
    Creates a random line at the origin
    """
    pp = (random_point_pair_at_origin()^einf).normal()
    return pp


def random_line():
    """
    Creates a random line
    """
    mv_a = random_euc_mv()
    mv_b = random_euc_mv()
    line_a = ((up(mv_a) ^ up(mv_b) ^ ninf)).normal()
    return line_a


def random_circle_at_origin():
    """
    Creates a random circle at the origin
    """
    mv_a = random_euc_mv()
    mv_r = random_euc_mv()
    r = generate_rotation_rotor(np.pi/2, mv_a, mv_r)
    mv_b = r*mv_a*~r
    mv_c = r * mv_b * ~r
    pp = (up(mv_a) ^ up(mv_b) ^ up(mv_c) ).normal()
    return pp


def random_circle():
    """
    Creates a random circle
    """
    mv_a = val_random_euc_mv()
    mv_b = val_random_euc_mv()
    mv_c = val_random_euc_mv()
    return layout.MultiVector(value=val_normalised(omt_func(omt_func(val_up(mv_a), val_up(mv_b)), val_up(mv_c))))


def random_sphere_at_origin():
    """
    Creates a random sphere at the origin
    """
    C = random_circle_at_origin()
    sphere = circle_to_sphere(C)
    return sphere


def random_sphere():
    """
    Creates a random sphere
    """
    mv_a = random_euc_mv()
    mv_b = random_euc_mv()
    mv_c = random_euc_mv()
    mv_d = random_euc_mv()
    sphere = ((up(mv_a) ^ up(mv_b) ^ up(mv_c) ^ up(mv_d))).normal()
    return sphere


def random_plane_at_origin():
    """
    Creates a random plane at the origin
    """
    c = random_circle_at_origin()
    plane = (c ^ einf).normal()
    return plane


def random_plane():
    """
    Creates a random plane
    """
    c = random_circle()
    plane = (c ^ ninf).normal()
    return plane


@numba.njit
def val_apply_rotor(mv_val, rotor_val):
    """ Applies rotor to multivector in a fast way - JITTED """
    return gmt_func(rotor_val, gmt_func(mv_val, adjoint_func(rotor_val)))


def apply_rotor(mv_in, rotor):
    """ Applies rotor to multivector in a fast way """
    return cf.MultiVector(layout, val_apply_rotor(mv_in.value, rotor.value))


@numba.njit
def val_apply_rotor_inv(mv_val, rotor_val, rotor_val_inv):
    """ Applies rotor to multivector in a fast way takes pre computed adjoint"""
    return gmt_func(rotor_val, gmt_func(mv_val, rotor_val_inv))


def apply_rotor_inv(mv_in, rotor, rotor_inv):
    """ Applies rotor to multivector in a fast way takes pre computed adjoint"""
    return cf.MultiVector(layout, val_apply_rotor_inv(mv_in.value, rotor.value, rotor_inv.value))


@numba.njit
def mult_with_ninf(mv):
    """ Convenience function for multiplication with ninf """
    return gmt_func(mv, ninf_val)


#@numba.njit
def val_convert_2D_polar_line_to_conformal_line(rho, theta):
    """ Converts a 2D polar line to a conformal line """
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 10000 * (-b))
    y1 = int(y0 + 10000 * (a))
    x2 = int(x0 - 10000 * (-b))
    y2 = int(y0 - 10000 * (a))
    p1_val = val_convert_2D_point_to_conformal(x1,y1)
    p2_val = val_convert_2D_point_to_conformal(x2,y2)
    line_val = omt_func(omt_func(p1_val,p2_val),ninf_val)
    line_val = line_val/abs(cf.MultiVector(layout,line_val))
    return line_val


def convert_2D_polar_line_to_conformal_line(rho, theta):
    """ Converts a 2D polar line to a conformal line """
    line_val = val_convert_2D_polar_line_to_conformal_line(rho, theta)
    return cf.MultiVector(layout, line_val)


@numba.njit
def val_up(mv_val):
    """ Fast jitted up mapping """
    temp = np.zeros(32)
    temp[0] = 0.5
    return mv_val - no_val + omt_func(temp, gmt_func(gmt_func(mv_val, mv_val), ninf_val))


def fast_up(mv):
    """ Fast up mapping """
    return cf.MultiVector(layout, val_up(mv.value))


@numba.njit
def val_normalInv(mv_val):
    """ A fast, jitted version of normalInv """
    Madjoint_val = adjoint_func(mv_val)
    MadjointM = gmt_func(Madjoint_val,mv_val)[0]
    return Madjoint_val / MadjointM


@numba.njit
def val_homo(mv_val):
    """ A fast, jitted version of homo() """
    return gmt_func(mv_val, val_normalInv(imt_func(-mv_val, ninf_val)))


@numba.njit
def val_down(mv_val):
    """ A fast, jitted version of down() """
    return gmt_func(omt_func(val_homo(mv_val), E0_val), E0_val)


def fast_down(mv):
    """ A fast version of down() """
    return cf.MultiVector(layout, val_down(mv.value))


def val_distance_point_to_line(point, line):
    """
    Returns the euclidean distance between a point and a line
    """
    return float(abs( cf.MultiVector(layout, omt_func(point,line) )  ))


@numba.njit
def val_convert_2D_point_to_conformal(x, y):
    """ Convert a 2D point to conformal """
    mv_val = np.zeros(32)
    mv_val[1] = x
    mv_val[2] = y
    return val_up(mv_val)


def convert_2D_point_to_conformal(x,y):
    """ Convert a 2D point to conformal """
    return cf.MultiVector(layout,val_convert_2D_point_to_conformal(x,y))


def distance_polar_line_to_euc_point_2d(rho, theta, x, y):
    """ Return the distance between a polar line and a euclidean point in 2D """
    point = val_convert_2D_point_to_conformal(x, y)
    line = val_convert_2D_polar_line_to_conformal_line(rho, theta)
    return val_distance_point_to_line(point, line)


dual_gmt_func = layout.gmt_func_generator(grades_a=[5], grades_b=[0, 1, 2, 3, 4, 5])


@numba.njit
def dual_func(a_val):
    """
    Fast dual
    """
    return dual_gmt_func(I5_val, a_val)


def fast_dual(a):
    """
    Fast dual
    """
    return cf.MultiVector(layout, dual_func(a.value))


class ConformalMVArray(cf.MVArray):
    """
    This class is for storing arrays of conformal multivectors
    """
    def up(self):
        """
        Up mapping
        """
        return v_up(self)

    def down(self):
        """
        Down mapping
        """
        return v_down(self)

    def dual(self):
        """
        Dualisation
        """
        return v_dual(self)

    def apply_rotor(self, R):
        """
        Application of a rotor
        """
        R_inv = ~R
        return v_apply_rotor_inv(self, R, R_inv)

    def apply_rotor_inv(self, R, R_inv):
        """
        Application of a rotor with precomputed inverse
        """
        return v_apply_rotor_inv(self, R, R_inv)

    @property
    def value(self):
        """
        Return an np array of the values of multivectors
        """
        return np.array([mv.value for mv in self])

    @staticmethod
    def from_value_array(value_array):
        """
        Constructs an array of mvs from a value array
        """
        return ConformalMVArray(v_new_mv(value_array))

v_dual = np.vectorize(fast_dual, otypes=[ConformalMVArray])
v_new_mv = np.vectorize(lambda v: cf.MultiVector(layout, v), otypes=[ConformalMVArray], signature='(n)->()')
v_up = np.vectorize(fast_up, otypes=[ConformalMVArray])
v_down = np.vectorize(fast_down, otypes=[ConformalMVArray])
v_apply_rotor_inv = np.vectorize(apply_rotor_inv, otypes=[ConformalMVArray])
v_meet = np.vectorize(meet, otypes=[ConformalMVArray], signature='(),()->()')

