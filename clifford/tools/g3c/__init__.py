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
    val_intersect_line_and_plane_to_point
    quaternion_and_vector_to_rotor
    get_center_from_sphere
    get_radius_from_sphere
    point_pair_to_end_points
    val_point_pair_to_end_points
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
    midpoint_between_lines
    val_midpoint_between_lines
    midpoint_of_line_cluster
    val_midpoint_of_line_cluster
    val_midpoint_of_line_cluster_grad
    get_line_intersection
    val_get_line_intersection
    project_points_to_plane
    project_points_to_sphere
    project_points_to_circle
    project_points_to_line

Misc
--------------------

.. autosummary::
    :toctree: generated/


    meet_val
    meet
    normalise_n_minus_1
    val_normalise_n_minus_1
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
    get_line_reflection_matrix
    val_get_line_reflection_matrix
    val_truncated_get_line_reflection_matrix
    interpret_multivector_as_object
    normalise_TR_to_unit_T
    scale_TR_translation


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
    n_th_rotor_root
    interp_objects_root
    general_object_interpolation
    average_objects
    val_average_objects_with_weights
    val_average_objects
    rotor_between_objects
    val_rotor_between_objects_root
    val_rotor_between_objects_explicit
    calculate_S_over_mu
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
from clifford import grades_present, NUMBA_PARALLEL, MVArray
from scipy.interpolate import interp1d

try:
    from pyganja import draw
    pyganja_available = True
except ImportError:
    pyganja_available = False


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
eo_val = eo.value

unit_scalar_mv = 1.0 + 0.0*e1
unit_scalar_mv_val = unit_scalar_mv.value

adjoint_func = layout.adjoint_func
gmt_func = layout.gmt_func
omt_func = layout.omt_func
imt_func = layout.imt_func

epsilon = 10*10**(-6)


mask0 = layout.get_grade_projection_matrix(0)
mask1 = layout.get_grade_projection_matrix(1)
mask2 = layout.get_grade_projection_matrix(2)
mask3 = layout.get_grade_projection_matrix(3)
mask4 = layout.get_grade_projection_matrix(4)
mask5 = layout.get_grade_projection_matrix(5)
mask_2minus4 = mask2 - mask4


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
    g_pres = grades_present(mv, 0.00000001)
    if len(g_pres) != 1:  # Definitely not a blade
        return -1
    grade, = g_pres
    if grade == 1:
        # This can now either be a euc mv or a conformal point or something else
        if mv(e123) == mv:
            return 1  # Euclidean point
        elif np.sum(np.abs((mv**2).value)) < 0.00000001:
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
            if abs(((mv*I5)|no)[0]) > epsilon:
                return 7  # Plane
            else:
                return 6  # Sphere
    else:
        return -1


@numba.njit
def val_sphere_line_intersect(s, l):
    """
    Checks for intersection between a sphere and a line
    """
    mv = meet_val(s, l)
    return imt_func(mv, mv)[0] > 0


def sphere_line_intersect(s, l):
    """
    Checks for intersection between a sphere and a line
    """
    return val_sphere_line_intersect(s.value, l.value)


def sphere_in_sphere(S1, S2, tolerance=10**-6):
    """
    Checks if one sphere is inside the other
    (S1|S2)[0] < -1
    """
    return (unsign_sphere(S1)|unsign_sphere(S2))[0] <= -1 + tolerance


def sphere_beyond_plane(sphere, plane):
    """
    Check if the sphere is fully beyond the plane in the direction of
    the plane normal
    """
    snorm = unsign_sphere(sphere)
    return (snorm|plane)[0] < -get_radius_from_sphere(snorm)


def sphere_behind_plane(sphere, plane):
    """
    Check if the sphere is fully behind the plane in the direction of
    the plane normal, ie the opposite of sphere_beyond_plane
    """
    snorm = unsign_sphere(sphere)
    return (snorm|plane)[0] > get_radius_from_sphere(snorm)


def point_beyond_plane(point, plane):
    """
    Check if the point is fully beyond the plane in the direction of
    the plane normal
    """
    return (point|(I5*plane))[0] < 0


def unsign_sphere(S):
    """
    Normalises the sign of a sphere
    """
    return -normalised(S * ((I5 * S) | einf)[0])


def val_unsign_sphere(S):
    """
    Normalises the sign of a sphere
    """
    return val_normalised(S*(-imt_func(dual_func(S), ninf_val)[0]))


def join_spheres(S1in, S2in):
    """
    Find the smallest sphere that encloses both spheres
    """
    s1 = unsign_sphere(S1in)
    s2 = unsign_sphere(S2in)
    L = normalised(((s1 * I5) ^ (s2 * I5) ^ einf)(3))
    pp1 = normalised(meet(s1, L)(2))
    pp2 = normalised(meet(s2, L)(2))
    p1 = point_pair_to_end_points(pp1)[0]
    p2 = point_pair_to_end_points(pp2)[1]
    if (p1|(s2*I5))[0] > 0.0:
        opt_sphere = s2(4)
    elif (p2|(s1*I5))[0] > 0.0:
        opt_sphere = s1(4)
    else:
        p12 = p1 ^ p2
        L2 = normalised(p12 * (p12 ^ einf))
        opt_sphere = (L2*I5)(4)
    return unsign_sphere(opt_sphere)


def enclosing_sphere(spheres):
    """
    For a given set of spheres this finds a sphere that encloses all of them
    Tries to find the smallest one it can
    """
    nspheres = len(spheres)
    if nspheres == 1:
        return spheres[0]
    elif nspheres == 2:
        return join_spheres(spheres[0], spheres[1])
    mins = spheres[0]
    for i in range(1, nspheres):
        mins = join_spheres(mins, spheres[i])
    return mins


def project_points_to_plane(point_list, plane):
    """
    Takes a load of points and projects them onto a plane
    """
    projected_list = []
    for point in point_list:
        proj_point = ((point|plane)*plane)
        proj_point = normalise_n_minus_1((proj_point*einf*proj_point)(1))
        projected_list.append(proj_point)
    return projected_list


def project_points_to_sphere(point_list, sphere, closest=True):
    """
    Takes a load of points and projects them onto a sphere
    """
    if closest:
        point_index = 1
    else:
        point_index = 0
    projected_list = []
    C = sphere*einf*sphere
    for point in point_list:
        proj_point = point_pair_to_end_points(meet(normalised(point^C^einf), sphere))[point_index]
        projected_list.append(proj_point)
    return projected_list


def project_points_to_circle(point_list, circle, closest=True):
    """
    Takes a load of point and projects them onto a circle
    The closest flag determines if it should be the closest or furthest point on the circle
    """
    circle_plane = normalised(circle^einf)
    planar_points = project_points_to_plane(point_list, circle_plane)
    circle_points = project_points_to_sphere(planar_points, -circle*circle_plane*I5, closest=closest)
    return circle_points


def project_points_to_line(point_list, line):
    """
    Takes a load of points and projects them onto a line
    """
    projected_list = []
    for point in point_list:
        pp = point|line
        proj_point = (pp*einf*pp)(1)
        projected_list.append(proj_point)
    return projected_list


def closest_points_on_circles(C1, C2, niterations=20):
    """
    Given two circles C1 and C2 this calculates the closest
    points on each of them to the other
    """
    cav = average_objects([C1, C2])
    cav2 = average_objects([C1, -C2])
    PP = meet(cav, cav2^einf).normal()
    P_list = point_pair_to_end_points(PP)
    dmin = np.inf
    for Ptest in P_list:
        d = -(project_points_to_circle([Ptest], C1)[0](1)|project_points_to_circle([Ptest], C2)[0](1))[0]
        if d < dmin:
            dmin = d
            P2 = Ptest
    P1 = project_points_to_circle([P2], C1)[0](1)
    P2 = project_points_to_circle([P1], C2)[0](1)
    for i in range(niterations):
        P1 = project_points_to_circle([P2], C1)[0](1)
        P2 = project_points_to_circle([P1], C2)[0](1)
    return P1, P2


def closest_points_circle_line(C, L, niterations=20):
    """
    Given a circle C and line L this calculates the closest
    points on each of them to the other
    """
    cav = average_objects([C, L])
    cav2 = average_objects([C, -L])
    PP = meet(cav, cav2^einf).normal()
    P_list = point_pair_to_end_points(PP)
    dmin = np.inf
    for Ptest in P_list:
        d = -(project_points_to_circle([Ptest], C)[0](1)|project_points_to_line([Ptest], L)[0](1))[0]
        if d < dmin:
            dmin = d
            P2 = Ptest
    P1 = project_points_to_circle([P2], C)[0](1)
    P2 = project_points_to_line([P1], L)[0](1)
    for i in range(niterations):
        P1 = project_points_to_circle([P2], C)[0](1)
        P2 = project_points_to_line([P1], L)[0](1)
    return P1, P2


def furthest_points_on_circles(C1, C2, niterations=20):
    """
    Given two circles C1 and C2 this calculates the closest
    points on each of them to the other
    """
    P2 = random_conformal_point()
    P1 = project_points_to_circle([P2], C1, closest=False)[0](1)
    P2 = project_points_to_circle([P1], C2, closest=False)[0](1)
    for i in range(niterations):
        P1 = project_points_to_circle([P2], C1, closest=False)[0](1)
        P2 = project_points_to_circle([P1], C2, closest=False)[0](1)
    return P1, P2


def normalise_TR_to_unit_T(TR):
    """
    Takes in a TR rotor
    extracts the R and T
    normalises the T to unit displacement magnitude
    rebuilds the TR rotor with the new displacement rotor
    returns the new TR and the original length of the T rotor
    """
    R_only = TR(e123)
    T_only = (TR*~R_only).normal()
    t = -2*(T_only|no)
    scale = abs(t)
    t = t/scale
    new_TR = (generate_translation_rotor(t)*R_only).normal()
    return new_TR, scale


def scale_TR_translation(TR, scale):
    """
    Takes in a TR rotor and a scale
    extracts the R and T
    scales the T displacement magnitude by scale
    rebuilds the TR rotor with the new displacement rotor
    returns the new TR rotor
    """
    R_only = TR(e123)
    T_only = (TR*~R_only).normal()
    t = -2*(T_only|no)*scale
    new_TR = (generate_translation_rotor(t)*R_only).normal()
    return new_TR


def left_gmt_generator(mt=layout.gmt):
    # unpack for numba
    k_list, l_list, m_list = mt.coords
    mult_table_vals = mt.data
    gaDims = mt.shape[1]
    val_get_left_gmt_matrix = cf._numba_val_get_left_gmt_matrix

    @numba.njit
    def get_left_gmt(x_val):
        return val_get_left_gmt_matrix(
            x_val, k_list, l_list, m_list, mult_table_vals, gaDims)
    return get_left_gmt


def right_gmt_generator(mt=layout.gmt):
    return left_gmt_generator(mt.T)


get_left_gmt_matrix = left_gmt_generator()
get_right_gmt_matrix = right_gmt_generator()


def get_line_reflection_matrix(lines, n_power=1):
    """
    Generates the matrix that sums the reflection of a point in many lines
    """
    mat2solve = np.zeros((32, 32), dtype=np.float64)
    for Li in lines:
        LiMat = get_left_gmt_matrix(Li.value)
        tmat = np.matmul(np.matmul(LiMat, mask_2minus4), LiMat)
        mat2solve += tmat
    mat = np.matmul(mask1, mat2solve)
    if n_power != 1:
        mat = np.linalg.matrix_power(mat, n_power)
    return mat


@numba.njit
def val_get_line_reflection_matrix(line_array, n_power):
    """
    Generates the matrix that sums the reflection of a point in many lines
    """
    mat2solve = np.zeros((32, 32), dtype=np.float64)
    for i in range(line_array.shape[0]):
        LiMat = get_left_gmt_matrix(line_array[i, :])
        tmat = np.matmul(np.matmul(LiMat, mask_2minus4), LiMat)
        mat2solve += tmat
    mat = np.matmul(mask1, mat2solve)
    if n_power != 1:
        mat = np.linalg.matrix_power(mat, n_power)
    return mat


@numba.njit
def val_truncated_get_line_reflection_matrix(line_array, n_power):
    """
    Generates the truncated matrix that sums the
    reflection of a point in many lines
    n_power should be 1 or a power of 2
    """
    mat2solve = np.zeros((32, 32), dtype=np.float64)
    for i in range(line_array.shape[0]):
        LiMat = get_left_gmt_matrix(line_array[i, :])
        tmat = np.matmul(np.matmul(LiMat, mask_2minus4), LiMat)
        mat2solve += tmat
    mat_val = np.matmul(mask1, mat2solve)
    mat_val = mat_val[1:6, 1:6].copy()/line_array.shape[0]
    if n_power != 1:
        c_pow = 1
        while c_pow < n_power:
            mat_val = np.matmul(mat_val, mat_val)
            c_pow = c_pow*2
    return mat_val


@numba.njit
def val_get_line_intersection(L3_val, Ldd_val):
    """
    Gets the point of intersection of two orthogonal lines that meet
    Xdd = Ldd*no*Ldd + no
    Xddd = L3*Xdd*L3
    Pd = 0.5*(Xdd+Xddd)
    P = -(Pd*ninf*Pd)(1)/(2*(Pd|einf)**2)[0]
    """
    Xdd = gmt_func(gmt_func(Ldd_val, no_val), Ldd_val) + no_val
    Xddd = gmt_func(gmt_func(L3_val, Xdd), L3_val)
    Pd = 0.5*(Xdd+Xddd)
    P = -gmt_func(gmt_func(Pd, ninf_val), Pd)
    imt_value = imt_func(Pd, ninf_val)
    P_denominator = 2*(gmt_func(imt_value, imt_value))[0]
    return project_val(P/P_denominator, 1)


def get_line_intersection(L3, Ldd):
    """
    Gets the point of intersection of two orthogonal lines that meet
    Xdd = Ldd*no*Ldd + no
    Xddd = L3*Xdd*L3
    Pd = 0.5*(Xdd+Xddd)
    P = -(Pd*ninf*Pd)(1)/(2*(Pd|einf)**2)[0]
    """
    return layout.MultiVector(value=val_get_line_intersection(L3.value, Ldd.value))


@numba.njit
def val_midpoint_between_lines(L1_val, L2_val):
    """
    Gets the point that is maximally close to both lines
    Hadfield and Lasenby AGACSE2018
    """
    L3 = val_normalised(L1_val + L2_val)
    Ldd = val_normalised(L1_val - L2_val)
    S = val_normalised(gmt_func(I5_val, val_get_line_intersection(L3, Ldd)))
    return val_normalise_n_minus_1(gmt_func(S, gmt_func(ninf_val, S)))


def midpoint_between_lines(L1, L2):
    """
    Gets the point that is maximally close to both lines
    Hadfield and Lasenby AGACSE2018
    """
    return layout.MultiVector(value=val_midpoint_between_lines(L1.value, L2.value))


def midpoint_of_line_cluster(line_cluster):
    """
    Gets a center point of a line cluster
    Hadfield and Lasenby AGACSE2018
    """
    return layout.MultiVector(value=val_midpoint_of_line_cluster(MVArray(line_cluster).value))


@numba.njit
def val_midpoint_of_line_cluster(array_line_cluster):
    """
    Gets a center point of a line cluster
    Hadfield and Lasenby AGACSE2018
    """
    # Allocate some space for our finished matrix
    accumulator_matrix = np.zeros((32, 32))

    # Loop over our lines and construct the matrix
    for i in range(array_line_cluster.shape[0]):
        # Get the line as a left gmt matrix
        L_i_l = get_left_gmt_matrix(array_line_cluster[i, :])
        # Get the line as a right gmt matrix
        L_i_r = get_right_gmt_matrix(array_line_cluster[i, :])
        # Multiply and add
        accumulator_matrix += L_i_r @ L_i_l

    # Raise the matrix to a very high power
    power_mat = np.linalg.matrix_power(accumulator_matrix / array_line_cluster.shape[0], 256)

    # Get a point that lies on the first line as an approximation to the e-vector
    pp_val = imt_func(array_line_cluster[0, :], eo_val)
    p_start = val_normalise_n_minus_1(project_val(gmt_func(gmt_func(pp_val, ninf_val), pp_val), 1))

    # Apply the matrix
    p_end = project_val(power_mat @ p_start, 1)

    # Remove any junk that has come along with it
    final_point = val_normalise_n_minus_1(project_val(gmt_func(gmt_func(p_end, ninf_val), p_end), 1))
    return final_point


@numba.njit(parallel=NUMBA_PARALLEL)
def val_midpoint_of_line_cluster_grad(array_line_cluster):
    """
    Gets an approximate center point of a line cluster
    Hadfield and Lasenby AGACSE2018
    """
    average_line = val_average_objects(array_line_cluster)
    val_point_track = np.zeros(32)
    for i in range(array_line_cluster.shape[0]):
        p = val_midpoint_between_lines(average_line, array_line_cluster[i, :])
        val_point_track += p
    S = gmt_func(I5_val, val_point_track)
    center_point = val_normalise_n_minus_1(project_val(gmt_func(S, gmt_func(ninf_val, S)), 1))
    # Take a derivative of the cost function at this point
    grad = np.zeros(32)
    for i in range(array_line_cluster.shape[0]):
        l_val = array_line_cluster[i, :]
        grad += (gmt_func(gmt_func(l_val, center_point), l_val))
    grad = val_normalise_n_minus_1(project_val(grad, 1))
    s_val = gmt_func(I5_val, project_val(center_point + grad, 1))
    center_point = val_normalise_n_minus_1(gmt_func(gmt_func(s_val, ninf_val), s_val))
    return center_point


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
    return [GAcentre, GAnormal, radius]


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
    return [p, mhat]


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


def generate_n_clusters(object_generator, n_clusters, n_objects_per_cluster):
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
    return layout.MultiVector(value=val_generate_translation_rotor(euc_vector_a.value))


@numba.njit
def val_generate_translation_rotor(euc_vector_a):
    """
    Generates a rotor that translates objects along the euclidean vector euc_vector_a
    """
    T = gmt_func(ninf_val, euc_vector_a) / 2
    T[0] += 1
    return T


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


@numba.njit
def val_intersect_line_and_plane_to_point(line_val, plane_val):
    """
    Returns the point at the intersection of a line and plane
    """
    m = meet_val(line_val, plane_val)
    if gmt_func(m, m)[0] < 0.000001:
        return np.array([-1.])
    output = np.zeros(32)
    A = val_normalised(m)
    if A[15] < 0:
        output[1] = A[8]
        output[2] = A[11]
        output[3] = A[14]
    else:
        output[1] = -A[8]
        output[2] = -A[11]
        output[3] = -A[14]
    output = val_up(output)
    return output


def intersect_line_and_plane_to_point(line, plane):
    """
    Returns the point at the intersection of a line and plane
    If there is no intersection it returns None
    """
    ans = val_intersect_line_and_plane_to_point(line.value, plane.value)
    if ans[0] == -1.:
        return None
    return layout.MultiVector(value=ans)


@numba.njit
def val_normalise_n_minus_1(mv_val):
    """
    Normalises a conformal point so that it has an inner product of -1 with einf
    """
    scale = imt_func(mv_val, ninf_val)[0]
    if scale != 0.0:
        return -mv_val/scale
    else:
        raise ZeroDivisionError('Multivector has 0 einf component')


def normalise_n_minus_1(mv):
    """
    Normalises a conformal point so that it has an inner product of -1 with einf
    """
    return layout.MultiVector(value=val_normalise_n_minus_1(mv.value))


def quaternion_and_vector_to_rotor(quaternion, vector):
    """
    Takes in a quaternion and a vector and returns a conformal rotor that
    implements the transformation
    """
    rotation = quaternion_to_rotor(quaternion)(0, 2, 4)
    translation = generate_translation_rotor(vector[0] * e1 + vector[1] * e2 + vector[2] * e3)(0, 2, 4)
    return (translation * rotation).normal()


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


@numba.njit
def val_point_pair_to_end_points(T):
    """
    Extracts the end points of a point pair bivector
    """
    beta = np.sqrt(abs(gmt_func(T, T)[0]))
    F = T / beta
    P = 0.5*F
    P[0] += 0.5
    P_twiddle = -0.5*F
    P_twiddle[0] += 0.5
    A = val_normalise_n_minus_1(-gmt_func(P_twiddle, imt_func(T, ninf_val)))
    B = val_normalise_n_minus_1(gmt_func(P, imt_func(T, ninf_val)))
    output = np.zeros((2, 32))
    output[0, :] = A
    output[1, :] = B
    return output


def point_pair_to_end_points(T):
    """
    Extracts the end points of a point pair bivector
    """
    output = val_point_pair_to_end_points(T.value)
    A = layout.MultiVector(value=output[0, :])
    B = layout.MultiVector(value=output[1, :])
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
    sqrd_ans = sigma_val[0] ** 2 - gmt_func(sigma_4, sigma_4)[0]
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
    """ The general case of the root of a grade 0, 4 multivector """
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
    output = np.zeros((2, 32))
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
    C3 = normalised(neg_twiddle_root(C)[0])
    if cf.grade_obj(C1, 0.00001) != cf.grade_obj(C3, 0.00001):
        raise ValueError('Created object is not same grade')
    return C3


def general_object_interpolation(object_alpha_array, object_list, new_alpha_array, kind='linear'):
    """
    Hadfield and Lasenby, Direct Linear Interpolation of Geometric Objects, AGACSE2018
    This is a general interpolation through the
    """
    obj_array = np.transpose(MVArray(object_list).value)
    f = interp1d(object_alpha_array, obj_array, kind=kind)
    new_value_array = np.transpose(f(new_alpha_array))
    new_conf_array = MVArray.from_value_array(layout, new_value_array)
    return [normalised(neg_twiddle_root(C)[0]) for C in new_conf_array]


@numba.njit
def val_average_objects_with_weights(obj_array, weights_array):
    """
    Hadfield and Lasenby, Direct Linear Interpolation of Geometric Objects, AGACSE2018
    Directly averages conformal objects
    Return a valid object from the addition result C
    """
    C_val = np.zeros(32)
    for i in range(obj_array.shape[0]):
        C_val += obj_array[i, :]*weights_array[i]
    C3 = val_normalised(neg_twiddle_root_val(C_val)[0, :])
    return C3


@numba.njit
def val_average_objects(obj_array):
    """
    Hadfield and Lasenby, Direct Linear Interpolation of Geometric Objects, AGACSE2018
    Directly averages conformal objects
    Return a valid object from the addition result C
    """
    C_val = np.zeros(32)
    for i in range(obj_array.shape[0]):
        C_val += obj_array[i, :]
    C_val = C_val / obj_array.shape[0]
    C3 = val_normalised(neg_twiddle_root_val(C_val)[0, :])
    return C3


def average_objects(obj_list, weights=[], check_grades=True):
    """
    Hadfield and Lasenby, Direct Linear Interpolation of Geometric Objects, AGACSE2018
    Directly averages conformal objects
    Return a valid object from the addition result C
    """
    if len(weights) == len(obj_list):
        C = sum([o * w for o, w in zip(obj_list, weights)])
    else:
        C = sum(obj_list) / len(obj_list)
    C3 = normalised(neg_twiddle_root(C)[0])
    if check_grades:
        if cf.grade_obj(obj_list[0], 0.00001) != cf.grade_obj(C3, 0.00001):
            raise ValueError('Created object is not same grade \n' + str(obj_list[0]) + '\n' + str(C3))
    return C3


def rotor_between_objects(X1, X2):
    """
    Lasenby and Hadfield AGACSE2018
    For any two conformal objects X1 and X2 this returns a rotor that takes X1 to X2
    Return a valid object from the addition result 1 + gamma*X2X1
    """
    return cf.MultiVector(
        layout,
        val_rotor_between_objects_root(val_normalised(X1.value),
                                       val_normalised(X2.value))
    )


def TRS_between_rounds(X1, X2):
    """
    Calculate the TRS rotor between any pair of rounds of the same grade
    Bring rounds to origin, line up carriers, calculate scale
    """
    T1 = generate_translation_rotor(-down((X1 * einf * X1)(1)))
    X1h = normalised(T1 * X1 * ~T1)
    T2 = generate_translation_rotor(-down((X2 * einf * X2)(1)))
    X2h = normalised(T2 * X2 * ~T2)
    X1f = normalised(X1h ^ einf)
    X2f = normalised(X2h ^ einf)
    Rc = rotor_between_objects(X1f, X2f)
    S = generate_dilation_rotor(get_radius_from_sphere(normalised(X2h*X2f*I5))/get_radius_from_sphere(normalised(X1h*X1f*I5)))
    return normalised((~T2)*S*Rc*T1)


def motor_between_rounds(X1, X2):
    """
    Calculate the motor between any pair of rounds of the same grade
    Line up the carriers, then line up the centers
    """
    return layout.MultiVector(value=val_motor_between_rounds(X1.value, X2.value))


@numba.njit
def val_motor_between_rounds(X1, X2):
    """
    Calculate the motor between any pair of rounds of the same grade
    Line up the carriers, then line up the centers

    Optimised form of this:

    R = rotor_between_objects(normalised(X1^einf), normalised(X2^einf))
    X3 = apply_rotor(X1, R)
    C1 = normalise_n_minus_1((X3 * einf * X3)(1)).value[1:4]
    C2 = normalise_n_minus_1((X2 * einf * X2)(1)).value[1:4]
    t = layout.MultiVector()
    t.value[1:4] = C2 - C1
    T = generate_translation_rotor(t)
    return normalised(T*R)
    """
    F1 = val_normalised(omt_func(X1, ninf_val))
    F2 = val_normalised(omt_func(X2, ninf_val))

    if np.abs(F1[31]) > 1E-5:
        # Its spheres we are dealing with
        R = unit_scalar_mv_val
        X3 = X1
    else:
        R = val_rotor_between_objects_root(F1, F2)
        X3 = val_apply_rotor(X1, R)

    C1 = val_normalise_n_minus_1(project_val(gmt_func(gmt_func(X3, ninf_val), X3), 1))
    C2 = val_normalise_n_minus_1(project_val(gmt_func(gmt_func(X2, ninf_val), X2), 1))

    t = np.zeros(32)
    t[1:4] = (C2 - C1)[1:4]
    T = val_generate_translation_rotor(t)
    return val_normalised(gmt_func(T, R))


@numba.njit
def val_motor_between_objects(X1, X2):
    """
    Calculates a motor that takes X1 to X2
    """
    carrier = omt_func(X1, ninf_val)
    if np.sum(np.abs(carrier)) < 1E-4:
        # They are flats
        return val_rotor_between_objects_root(X1, X2)
    else:
        # Rounds
        return val_motor_between_rounds(X1, X2)


def motor_between_objects(X1, X2):
    """
    Calculates a motor that takes X1 to X2
    """
    return layout.MultiVector(value=val_motor_between_objects(X1.value, X2.value))


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


I5eoval = (I5 * eo).value
biv3dmask = (e12+e13+e23).value


@numba.njit
def val_rotor_between_objects_root(X1, X2):
    """
    Lasenby and Hadfield AGACSE2018
    For any two conformal objects X1 and X2 this returns a rotor that takes X1 to X2
    Uses the square root of rotors for efficiency and numerical stability
    """
    X21 = gmt_func(X2, X1)
    X12 = gmt_func(X1, X2)
    gamma = gmt_func(X1, X1)[0]
    if gamma > 0:
        C_val = gamma*gmt_func(X2, X1)
        C_val[0] += 1
        if abs(C_val[0]) < 1E-6:
            R = val_normalised(project_val(gmt_func(I5eoval, X21), 2))
            return val_normalised(gmt_func(R, val_rotor_between_objects_root(X1, -X2)))
        return val_normalised(pos_twiddle_root_val(C_val)[0, :])
    else:
        C_val = -X21
        C_val[0] += 1
        if abs(C_val[0]) < 1E-6:
            R = project_val(gmt_func(I5eoval, X21), 2)
            R = val_normalised(project_val(gmt_func(R, biv3dmask), 2))
            R2 = val_normalised(val_rotor_between_objects_root(val_apply_rotor(X1, R), X2))
            return val_normalised(gmt_func(R2, R))
        else:
            return val_normalised(C_val)


@numba.njit
def val_rotor_between_objects_explicit(X1, X2):
    """
    Lasenby and Hadfield AGACSE2018
    For any two conformal objects X1 and X2 this returns a rotor that takes X1 to X2

    Implements an optimised version of::

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
        beta_sqrd = 1 / (2*root_mu + 2*K_val[0])
        beta = np.sqrt(beta_sqrd)

        temp_1 = beta * K_val_4
        temp_1[0] = temp_1[0] - (1 / (2 * beta))

        temp_2 = gamma1*gmt_func(X2, X1)
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
    r"""
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
    """ Creates a standard line at the origin """
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
    pp = (up(mv_a) ^ up(mv_b) ^ up(mv_c)).normal()
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


# @numba.njit
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
    p1_val = val_convert_2D_point_to_conformal(x1, y1)
    p2_val = val_convert_2D_point_to_conformal(x2, y2)
    line_val = omt_func(omt_func(p1_val, p2_val), ninf_val)
    line_val = line_val/abs(cf.MultiVector(layout, line_val))
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
    MadjointM = gmt_func(Madjoint_val, mv_val)[0]
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
    return float(abs(cf.MultiVector(layout, omt_func(point, line))))


@numba.njit
def val_convert_2D_point_to_conformal(x, y):
    """ Convert a 2D point to conformal """
    mv_val = np.zeros(32)
    mv_val[1] = x
    mv_val[2] = y
    return val_up(mv_val)


def convert_2D_point_to_conformal(x, y):
    """ Convert a 2D point to conformal """
    return cf.MultiVector(layout, val_convert_2D_point_to_conformal(x, y))


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
    def draw(self):
        '''
        display mvarray using a given visualization backend

        currently supports pyganja.
        '''
        if pyganja_available:
            return draw([mv for mv in self])
        else:
            pass

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
