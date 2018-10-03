
import numba
import numpy as np
from clifford import grade_obj
from clifford.g3c import *
import clifford as cf
from . import rotor_between_objects, rotor_between_lines, val_normalised

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


gradeList = layout.gradeList
grade_obj_func = cf.grade_obj_func
def val_object_cost_function(obj_a_val, obj_b_val):
    """
    Evaluates the rotor cost function between two objects
    """
    grade_a = grade_obj_func(obj_a_val, gradeList, 0.00001)
    grade_b = grade_obj_func(obj_b_val, gradeList, 0.00001)
    if grade_a != grade_b:
        return np.finfo(float).max
    else:
        R = rotor_between_objects(cf.MultiVector(layout, obj_a_val), cf.MultiVector(layout, obj_b_val))
        return np.abs(val_rotor_cost_sparse(R.value))


def object_cost_function(obj_a,obj_b):
    """
    Evaluates the rotor cost function between two objects
    """
    if grade_obj(obj_a, 0.00001) != grade_obj(obj_b, 0.00001):
        return np.finfo(float).max
    R = rotor_between_objects(obj_a.normal(), obj_b.normal())
    return np.abs(val_rotor_cost_sparse(R.value))


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


def object_set_cost_matrix(object_set_a, object_set_b, object_type='generic'):
    """
    Evaluates the rotor cost matrix between two sets of objects
    """
    matrix = np.zeros((len(object_set_a), len(object_set_b)))
    if object_type == 'lines':
        for a_ind, l_a in enumerate(object_set_a):
            for b_ind, l_b in enumerate(object_set_b):
                matrix[a_ind, b_ind] = line_cost_function(l_a, l_b)
    else:
        for a_ind, l_a in enumerate(object_set_a):
            for b_ind, l_b in enumerate(object_set_b):
                matrix[a_ind, b_ind] = object_cost_function(l_a, l_b)
    return matrix


def object_set_cost_matrix_sum(object_set_a, object_set_b, object_type='generic'):
    """
    Evaluates the sum of all elements in the rotor cost matrix between two sets of objects
    """
    if (len(object_set_a) == 0) or (len(object_set_b) == 0):
        return np.inf
    return np.sum(object_set_cost_matrix(object_set_a, object_set_b, object_type))


def line_cost_function(obj_a,obj_b):
    """
    Evaluates the rotor cost function between two objects
    """
    if grade_obj(obj_a, 0.00001) != grade_obj(obj_b, 0.00001):
        return np.finfo(float).max
    R = rotor_between_lines(obj_a, obj_b)
    return np.abs(val_rotor_cost_sparse(R.value))
