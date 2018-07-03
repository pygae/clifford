import random
from scipy import e
import numpy as np
from scipy.optimize import minimize

from .rotor_parameterisation import rotorconversion
from . import rotor_between_objects, apply_rotor, square_roots_of_rotor
from clifford.g3c import *
from clifford.tools import orthoFrames2Verser as cartan
from .cost_functions import object_set_cost_sum, rotor_cost


I5 = e12345
imt_func = layout.imt_func
gmt_func = layout.gmt_func
adjoint_func = layout.adjoint_func
e4_val = e4.value
ninf_val = einf.value
e123inf =e123*einf


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


def cartans_lines(obj_list_a, obj_list_b):
    """ Performs the extended cartans algorithm as suggested by Alex Arsenovic """
    V_found,rs = cartan(A=obj_list_a,B=obj_list_b)
    theta = ((V_found*~V_found)*e1234)(0)
    V_found = e**(-theta/2*e123inf)*V_found
    return V_found


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



