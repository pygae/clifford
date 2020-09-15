
from . import *
from clifford import NUMBA_PARALLEL, MVArray
import itertools
from .rotor_parameterisation import general_logarithm

imt_func = layout.imt_func
gmt_func = layout.gmt_func
adjoint_func = layout.adjoint_func


sparse_cost_imt = layout.imt_func_generator(grades_a=[0, 2, 4], grades_b=[1])
sparse_cost_gmt = layout.gmt_func_generator(grades_a=[0, 2, 4], grades_b=[0, 2, 4])

gradeList = np.array(layout.gradeList)
grade_obj_func = cf.grade_obj_func
max_float = np.finfo(float).max


def check_p(Q):
    """
    For leo dorsts check product
    """
    return Q(0, 1, 3) - Q(2, 4, 5)


def check_p_cost(P, Q):
    """
    For leo dorsts check product cost
    """
    return (P|check_p(Q))[()]


def point_to_line_cluster_distance(point, line_cluster):
    """
    Distance between a single point and a cluster of lines
    """
    return val_point_to_line_cluster_distance(point.value, np.array([l.value for l in line_cluster]))


@numba.njit
def val_point_to_line_cluster_distance(point_val, line_cluster_array):
    """
    Distance between a single point and a cluster of lines
    """
    error_val = 0.0
    for i in range(line_cluster_array.shape[0]):
        l_val = line_cluster_array[i, :]
        error_val -= imt_func(point_val, (gmt_func(gmt_func(l_val, point_val), l_val)))[0]
    return error_val


def midpoint_and_error_of_line_cluster_eig(line_cluster):
    """
    Gets an approximate center point of a line cluster
    as well as an estimate of the error
    Uses the eigenvalue and explicit derivatives
    """
    line_cluster_array = np.array([l.value for l in line_cluster], dtype=np.float64)
    mat2solve = val_truncated_get_line_reflection_matrix(line_cluster_array, 128)
    start = imt_func(no.value, sum(l.value for l in line_cluster))
    start = gmt_func(gmt_func(start, ninf.value), start)[1:6]

    point_val = np.zeros(32)
    point_val[1:6] = np.matmul(mat2solve, start)
    new_mv = layout.MultiVector(point_val)
    new_mv = normalise_n_minus_1((new_mv * einf * new_mv)(1))
    return new_mv, val_point_to_line_cluster_distance(new_mv.value, line_cluster_array)


def midpoint_and_error_of_line_cluster_svd(line_cluster):
    """
    Gets an approximate center point of a line cluster
    as well as an estimate of the error
    Uses the SVD and explicit derivatives
    """
    line_cluster_array = np.array([l.value for l in line_cluster])
    mat2solve = get_line_reflection_matrix(line_cluster)

    grade_val = 1
    column_mask = np.array(layout.gradeList) == grade_val
    mat_test = mat2solve[:, column_mask][1:6]
    print(mat_test)
    w, v = np.linalg.eig(mat_test)

    point_val = np.zeros(32)
    point_val[np.array(layout.gradeList) == grade_val] = v[:, 1]
    new_mv = layout.MultiVector(point_val)
    # new_mv = normalise_n_minus_1(new_mv * einf * new_mv)
    new_point = normalise_n_minus_1(new_mv)  # up(down(new_mv) / 2)
    return new_point, val_point_to_line_cluster_distance(new_point.value, line_cluster_array)


def midpoint_and_error_of_line_cluster(line_cluster):
    """
    Gets an approximate center point of a line cluster
    as well as an estimate of the error

    From :cite:`rotor-between`.
    """
    line_cluster_array = np.array([l.value for l in line_cluster])
    cp_val = val_midpoint_of_line_cluster(line_cluster_array)
    return layout.MultiVector(cp_val), val_point_to_line_cluster_distance(cp_val, line_cluster_array)


def midpoint_and_error_of_line_cluster_grad(line_cluster):
    """
    Gets an approximate center point of a line cluster
    as well as an estimate of the error

    From :cite:`rotor-between`.
    """
    line_cluster_array = np.array([l.value for l in line_cluster])
    cp_val = val_midpoint_of_line_cluster_grad(line_cluster_array)
    return layout.MultiVector(cp_val), val_point_to_line_cluster_distance(cp_val, line_cluster_array)


def line_plane_cost(line, plane):
    """
    A cost function for a line and a plane
    """
    P = normalised((line|plane)*I5)
    L = normalised(meet(P, plane))
    return line_cost_function(L, line)


def midline_and_error_of_plane_cluster(plane_cluster):
    """
    Finds the line that is the best intersection of the planes
    """
    plane_perms = itertools.permutations(plane_cluster, 2)
    line_list = [normalised(meet(pp[0], pp[1])(3)) for pp in plane_perms]

    ref_line = line_list[0]
    line_sum = 0.0 * e1
    for l in line_list:
        if (ref_line | l)[()] > 0:
            line_sum += l
        else:
            line_sum -= l
    line_average = average_objects([line_sum(3)])
    cost_val = 0.0
    for plane in plane_cluster:
        cost_val += line_plane_cost(line_average, plane)
    return line_average, cost_val


@numba.njit
def val_rotor_cost_sparse(R_val):
    """
    Evaluates Eivind Eiede's cost function of a rotor
    """
    rotation_val = R_val.copy()
    rotation_val[0] -= 1
    translation_val = sparse_cost_imt(R_val, e4.value)
    a = abs(float(sparse_cost_gmt(rotation_val, adjoint_func(rotation_val))[0]))
    b = abs(float(gmt_func(translation_val, adjoint_func(translation_val))[0]))
    return a + b


def rotor_cost(R):
    """
    Evaluates Eivind Eiede's cost function of a rotor
    """
    return val_rotor_cost_sparse(R.value)


@numba.njit
def val_object_cost_function(obj_a_val, obj_b_val, motor=False):
    """
    Evaluates the rotor cost function between two objects
    """
    grade_a = grade_obj_func(obj_a_val, gradeList, 0.00001)
    grade_b = grade_obj_func(obj_b_val, gradeList, 0.00001)
    if grade_a != grade_b:
        return max_float
    else:
        if motor:
            R_val = val_motor_between_objects(obj_a_val, obj_b_val)
        else:
            R_val = val_rotor_between_objects_root(obj_a_val, obj_b_val)
        return np.abs(val_rotor_cost_sparse(R_val))


def alt_rotor_cost(V):
    """
    An alternate rotor cost for TRS rotors
    """
    logV = general_logarithm(V)
    scale_cost = np.abs(-2*logV[e45])
    scalefac = np.e**(-2*logV[e45])
    R = logV(e123)*e123
    rotation_cost = abs((R*~R)[()])
    translation_cost = scalefac*abs((logV - logV[e45]*e45 - logV(e123))|eo)
    return rotation_cost + scale_cost + translation_cost


def object_cost_function(obj_a, obj_b, motor=False, symmetric=False):
    """
    Evaluates the rotor cost function between two objects
    """
    if symmetric:
        return min([val_object_cost_function(obj_a.value, obj_b.value, motor=motor),
                    val_object_cost_function(obj_a.value, -obj_b.value, motor=motor)])
    return val_object_cost_function(obj_a.value, obj_b.value, motor=motor)


def object_set_log_cost_sum(object_set_a, object_set_b, object_type='generic'):
    """
    Evaluates the rotor cost function between two sets of objects
    """
    sum_val = 0.0
    if object_type == 'lines':
        for a, b in zip(object_set_a, object_set_b):
            sum_val += np.log1p(line_cost_function(a, b))
    else:
        for a, b in zip(object_set_a, object_set_b):
            sum_val += np.log1p(object_cost_function(a, b))
    return sum_val


def object_set_cost_sum(object_set_a, object_set_b,
                        object_type='generic', motor=False,
                        symmetric=False):
    """
    Evaluates the rotor cost function between two sets of objects
    """
    sum_val = 0.0
    if object_type == 'lines':
        for a, b in zip(object_set_a, object_set_b):
            sum_val += line_cost_function(a, b)
    else:
        for a, b in zip(object_set_a, object_set_b):
            sum_val += object_cost_function(a, b, motor=motor, symmetric=symmetric)
    return sum_val


@numba.njit(parallel=NUMBA_PARALLEL, nogil=True)
def val_object_set_cost_matrix(object_array_a, object_array_b):
    """
    Evaluates the rotor cost matrix between two sets of objects
    """
    matrix = np.zeros((object_array_a.shape[0], object_array_b.shape[0]))
    for a_ind in range(object_array_a.shape[0]):
        l_a = object_array_a[a_ind, :]
        for b_ind in range(object_array_b.shape[0]):
            l_b = object_array_b[b_ind, :]
            matrix[a_ind, b_ind] = val_object_cost_function(l_a, l_b)
    return matrix


@numba.njit(parallel=NUMBA_PARALLEL, nogil=True)
def val_line_set_cost_matrix(object_array_a, object_array_b):
    """
    Evaluates the rotor cost matrix between two sets of objects
    """
    matrix = np.zeros((object_array_a.shape[0], object_array_b.shape[0]))
    for a_ind in range(object_array_a.shape[0]):
        l_a = object_array_a[a_ind, :]
        for b_ind in range(object_array_b.shape[0]):
            l_b = object_array_b[b_ind, :]
            matrix[a_ind, b_ind] = val_line_cost_function(l_a, l_b)
    return matrix


def object_set_cost_matrix(object_set_a, object_set_b,
                           object_type="generic", symmetric=False):
    """
    Evaluates the rotor cost matrix between two sets of objects
    """
    object_array_a = MVArray(object_set_a).value
    object_array_b = MVArray(object_set_b).value
    if object_type == 'lines':
        ret_mat = val_line_set_cost_matrix(object_array_a, object_array_b)
        if symmetric:
            ret_mat = np.minimum(ret_mat, val_line_set_cost_matrix(object_array_a, -object_array_b))
        return ret_mat
    else:
        ret_mat = val_object_set_cost_matrix(object_array_a, object_array_b)
        if symmetric:
            ret_mat = np.minimum(ret_mat, val_object_set_cost_matrix(object_array_a, -object_array_b))
        return ret_mat


def object_set_cost_matrix_sum(object_set_a, object_set_b, object_type='generic'):
    """
    Evaluates the sum of all elements in the rotor cost matrix between two sets of objects
    """
    if (len(object_set_a) == 0) or (len(object_set_b) == 0):
        return np.inf
    return np.sum(object_set_cost_matrix(object_set_a, object_set_b, object_type))


@numba.njit
def val_line_cost_function(obj_a_val, obj_b_val):
    grade_a = grade_obj_func(obj_a_val, gradeList, 0.00001)
    grade_b = grade_obj_func(obj_b_val, gradeList, 0.00001)
    if grade_a != grade_b:
        return max_float
    else:
        if imt_func(obj_a_val, obj_b_val)[0] < 0:
            R_val = val_rotor_between_lines(obj_a_val, -obj_b_val)
        else:
            R_val = val_rotor_between_lines(obj_a_val, obj_b_val)
        return np.abs(val_rotor_cost_sparse(R_val))


def line_cost_function(obj_a, obj_b):
    """
    Evaluates the rotor cost function between two objects
    """
    return val_line_cost_function(obj_a.value, obj_b.value)
