
import numpy as np
import numba
from scipy.optimize import minimize
from clifford import get_mult_function, grade_obj
from . import rotor_between_objects, apply_rotor, val_exp, square_roots_of_rotor
from clifford.g3c import *
import clifford as cf
from clifford.tools import orthoFrames2Verser as cartan
from scipy import e,rand

imt_func = layout.imt_func
gmt_func = layout.gmt_func
adjoint_func = layout.adjoint_func
e4_val = e4.value
ninf_val = einf.value

sparse_cost_imt = get_mult_function(layout.imt, layout.gaDims, layout.gradeList, grades_a=[0, 2, 4], grades_b=[1])
sparse_cost_gmt = get_mult_function(layout.gmt, layout.gaDims, layout.gradeList, grades_a=[0, 2, 4], grades_b=[0, 2, 4])


@numba.njit
def val_vec_repr_to_bivector(x):
    """ Converts between the parameters of a bivector and the bivector itself """
    t_val = np.zeros(32)
    t_val[1] = x[0]
    t_val[2] = x[1]
    t_val[3] = x[2]
    B_val = gmt_func(t_val, ninf_val)
    B_val[6] += x[3]
    B_val[7] += x[4]
    B_val[10] += x[5]
    return B_val


@numba.njit
def val_rotorconversion(x):
    """ Converts between the parameters of a bivector and the rotor that it is generating """
    B_val = val_vec_repr_to_bivector(x)
    R_val = val_exp(B_val)
    return R_val


def rotorconversion(x):
    """ Converts between the parameters of a bivector and the rotor that it is generating """
    return cf.MultiVector(layout, val_rotorconversion(x))


@numba.njit
def val_rotor_cost_sparse(R_val):
    """ Evaluates Eivind Eiede's cost function of a rotor """
    rotation_val = R_val.copy()
    rotation_val[0] -= 1
    translation_val = sparse_cost_imt(R_val, e4_val)
    a = abs(float(sparse_cost_gmt(rotation_val, adjoint_func(rotation_val))[0]))
    b = abs(float(gmt_func(translation_val, adjoint_func(translation_val))[0]))
    return a + b


def rotor_cost(R):
    """ Evaluates Eivind Eiede's cost function of a rotor """
    return val_rotor_cost_sparse(R.value)


def object_cost_function(obj_a,obj_b):
    """ Evaluates the rotor cost function between two objects """
    if grade_obj(obj_a) != grade_obj(obj_b):
        return np.finfo(float).max
    R = rotor_between_objects(obj_a, obj_b)
    return np.abs(val_rotor_cost_sparse(R.value))


def object_set_log_cost_sum(object_set_a, object_set_b):
    """ Evaluates the rotor cost function between two sets of objects """
    sum_val = 0.0
    for a, b in zip(object_set_a, object_set_b):
        sum_val += np.log1p(object_cost_function(a, b))
    return sum_val


def object_set_cost_sum(object_set_a, object_set_b):
    """ Evaluates the rotor cost function between two sets of objects """
    sum_val = 0.0
    for a, b in zip(object_set_a, object_set_b):
        sum_val += object_cost_function(a, b)
    return sum_val


def estimate_rotor_objects(object_list_a, object_list_b, maxfev=20000, print_res=False):
    """
    Estimates the rotor that takes one set of objects to another
    TODO improve convergence for point pairs, maybe change optimisation framework
    """
    x0 = np.finfo(float).eps * np.random.rand(6)
    def minimisation_func(x):
        R = rotorconversion(x)
        object_set_a = [apply_rotor(l, R).normal() for l in object_list_a]
        return object_set_cost_sum(object_set_a, object_list_b)
    res = minimize(minimisation_func, x0, method='SLSQP', options={'ftol': 10.0 ** (-16), \
                                                                      'maxiter': 500, \
                                                                      'disp': False, \
                                                                      'maxls': 40})
    if print_res:
        print(res)
    res = minimize(minimisation_func, res.x, method='L-BFGS-B', options={'ftol':10.0**(-16), \
                                                                   'maxiter':500, \
                                                                   'disp': False, \
                                                                    'maxls':40})
    if print_res:
        print(res)
    rotor = rotorconversion(res.x)
    object_set_a = [apply_rotor(l, rotor).normal() for l in object_list_a]
    cost = object_set_cost_sum(object_set_a, object_list_b)
    return rotor, cost


e123inf =e123*einf
def cartans_lines(obj_list_a, obj_list_b):
    """ Performs the extended cartans algorithm as suggested by Alex Arsenovic """
    V_found,rs = cartan(A=obj_list_a,B=obj_list_b)
    theta = ((V_found*~V_found)*e1234)(0)
    V_found = e**(-theta/2*e123inf)*V_found
    return V_found


import random
def sequential_object_rotor_estimation(obj_list_a, obj_list_b, n_iterations=500, cost_tolerance=10*(10**-16), random_sequence=False):
    """
    Performs a sequential rotor update based on the rotors between individual objects
    Exits when the sum of the cost of rotor updates through the list is very small
    """
    R_total = 1.0 + 0.0*e1
    r_list = []
    for j in range(n_iterations):
        cost_sum = 0.0
        if random_sequence:
            indices = random.sample(range(len(obj_list_a)), len(obj_list_a))
        else:
            indices = range(len(obj_list_a))
        for i in indices:
            C1 = (apply_rotor(obj_list_a[i],R_total)).normal()
            C2 = obj_list_b[i]
            if abs(C1 + C2) < 0.0001:
                C1 = -C1
            rroot = (square_roots_of_rotor(rotor_between_objects(C1,C2))[0]).normal()
            r_list.append(rroot)
            R_total = (rroot*R_total).normal()
            cost_sum += rotor_cost(rroot)
        if cost_sum < cost_tolerance:
            exit_flag = 0
            return R_total, r_list, exit_flag
    exit_flag = 1
    return R_total, r_list, exit_flag


def sequential_object_rotor_estimation_convergence_detection(obj_list_a, obj_list_b, n_iterations=500, cost_tolerance=10*(10**-16), random_sequence=False):
    """
    Performs a sequential rotor update based on the rotors between individual objects
    Exits when a full rotation through all objects produces a very small update of rotor
    """
    R_total = 1.0 + 0.0*e1
    r_list = []
    for j in range(n_iterations):
        r_set = 1.0 + 0.0*e1
        if random_sequence:
            indices = random.sample(range(len(obj_list_a)), len(obj_list_a))
        else:
            indices = range(len(obj_list_a))
        for i in indices:
            C1 = (apply_rotor(obj_list_a[i],R_total)).normal()
            C2 = obj_list_b[i]
            if abs(C1 + C2) < 0.0001:
                C1 = -C1
            rroot = (square_roots_of_rotor(rotor_between_objects(C1,C2))[0]).normal()
            r_list.append(rroot)
            r_set = (rroot*r_set).normal()
            R_total = (rroot * R_total).normal()
        if rotor_cost(r_set) < cost_tolerance:
            exit_flag = 0
            return R_total, r_list, exit_flag
    exit_flag = 1
    return R_total, r_list, exit_flag



