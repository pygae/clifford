
import numba
import numpy as np
from clifford import grade_obj
from clifford.g3c import *
import clifford as cf
from . import rotor_between_objects, val_rotor_between_lines, \
    rotor_between_lines, val_normalised, val_rotor_between_objects_root, ConformalMVArray

imt_func = layout.imt_func
gmt_func = layout.gmt_func
adjoint_func = layout.adjoint_func
e4_val = e4.value
e5_val = e5.value
ninf_val = einf.value


sparse_cost_imt = layout.imt_func_generator(grades_a=[0, 2, 4], grades_b=[1])
sparse_cost_gmt = layout.gmt_func_generator(grades_a=[0, 2, 4], grades_b=[0, 2, 4])


@numba.njit
def val_rotor_cost_sparse(R_val):
    """
    Evaluates Eivind Eiede's cost function of a rotor
    """
    rotation_val = R_val.copy()
    rotation_val[0] -= 1
    translation_val = sparse_cost_imt(R_val, e4_val)
    a = abs(float(sparse_cost_gmt(rotation_val, adjoint_func(rotation_val))[0]))
    b = abs(float(gmt_func(translation_val, adjoint_func(translation_val))[0]))
    return a + b


def rotor_cost(R):
    """
    Evaluates Eivind Eiede's cost function of a rotor
    """
    return val_rotor_cost_sparse(R.value)


gradeList = np.array(layout.gradeList)
grade_obj_func = cf.grade_obj_func
max_float = np.finfo(float).max
@numba.njit
def val_object_cost_function(obj_a_val, obj_b_val):
    """
    Evaluates the rotor cost function between two objects
    """
    grade_a = grade_obj_func(obj_a_val, gradeList, 0.00001)
    grade_b = grade_obj_func(obj_b_val, gradeList, 0.00001)
    if grade_a != grade_b:
        return max_float
    else:
        R_val = val_rotor_between_objects_root(obj_a_val, obj_b_val)
        return np.abs(val_rotor_cost_sparse(R_val))


def object_cost_function(obj_a, obj_b):
    """
    Evaluates the rotor cost function between two objects
    """
    return val_object_cost_function(obj_a.value, obj_b.value)


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


def object_set_cost_sum(object_set_a, object_set_b, object_type='generic'):
    """
    Evaluates the rotor cost function between two sets of objects
    """
    sum_val = 0.0
    if object_type == 'lines':
        for a, b in zip(object_set_a, object_set_b):
            sum_val += line_cost_function(a, b)
    else:
        for a, b in zip(object_set_a, object_set_b):
            sum_val += object_cost_function(a, b)
    return sum_val


@numba.njit(parallel=True, nogil=True)
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


@numba.njit(parallel=True, nogil=True)
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


def object_set_cost_matrix(object_set_a, object_set_b, object_type='generic'):
    """
    Evaluates the rotor cost matrix between two sets of objects
    """
    object_array_a = ConformalMVArray(object_set_a).value
    object_array_b = ConformalMVArray(object_set_b).value
    if object_type == 'lines':
        return val_line_set_cost_matrix(object_array_a, object_array_b)
    else:
        return val_object_set_cost_matrix(object_array_a, object_array_b)



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
        R_val = val_rotor_between_lines(obj_a_val, obj_b_val)
        return np.abs(val_rotor_cost_sparse(R_val))


def line_cost_function(obj_a, obj_b):
    """
    Evaluates the rotor cost function between two objects
    """
    return val_line_cost_function(obj_a.value, obj_b.value)
