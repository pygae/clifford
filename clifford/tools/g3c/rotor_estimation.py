import random
from scipy import e
import numpy as np
import multiprocessing

from scipy.optimize import minimize

from .rotor_parameterisation import rotorconversion
from . import rotor_between_objects, apply_rotor, square_roots_of_rotor, rotor_between_lines, normalised
from clifford.g3c import *
from clifford.tools import orthoFrames2Verser as cartan
from .cost_functions import object_set_cost_sum, rotor_cost
from clifford import grade_obj

I5 = e12345
imt_func = layout.imt_func
gmt_func = layout.gmt_func
adjoint_func = layout.adjoint_func
e4_val = e4.value
ninf_val = einf.value
e123inf =e123*einf


def estimate_rotor_objects_subsample(reference_model, query_model, n_repeats, objects_per_sample,
                                     maxfev=20000, print_res=False, pool_size=1, object_type='generic'):
    """
    Estimates the rotor that takes one set of objects to another
    """
    if int(pool_size) < 2:
        min_cost = np.finfo(float).max
        min_rotor = 1.0 + 0.0 * e1
        for i in range(n_repeats):
            indices = random.sample(range(len(reference_model)), objects_per_sample)
            object_sample_reference = [reference_model[j] for j in indices]
            object_sample_query = [query_model[j] for j in indices]
            rotor, new_cost = estimate_rotor_objects(object_sample_reference, object_sample_query, object_type=object_type)
            if new_cost < min_cost:
                min_cost = new_cost
                min_rotor = rotor
            print('SAMPLE: ', i, '  cost  ', min_cost)
    elif int(pool_size) > 1:
        with multiprocessing.Pool(int(pool_size)) as pool_obj:
            object_sample_pairs = []
            for i in range(n_repeats):
                indices = random.sample(range(len(reference_model)), objects_per_sample)
                object_sample_reference = [reference_model[j] for j in indices]
                object_sample_query = [query_model[j] for j in indices]
                object_sample_pairs.append([object_sample_reference, object_sample_query])
            if object_type == 'lines':
                starmap_output = pool_obj.starmap(estimate_rotor_lines, object_sample_pairs)
            else:
                starmap_output = pool_obj.starmap(estimate_rotor_objects, object_sample_pairs)
            min_rotor, min_cost = min(starmap_output, key=lambda x: x[1])
            print('SAMPLE: ', n_repeats, '  cost  ', min_cost)
    else:
        raise ValueError('Pool size is not valid')
    return min_rotor, min_cost



def estimate_rotor_objects_subsample_sequential(reference_model, query_model, n_repeats, objects_per_sample,
                                     maxfev=20000, print_res=False, pool_size=1, object_type='generic'):
    """
    Estimates the rotor that takes one set of objects to another
    """
    min_cost = np.finfo(float).max
    min_rotor = 1.0 + 0.0 * e1
    for i in range(n_repeats):
        indices = random.sample(range(len(reference_model)), objects_per_sample)
        object_sample_reference = [reference_model[j] for j in indices]
        object_sample_query = [query_model[j] for j in indices]
        rotor, r_list, e_flag = sequential_object_rotor_estimation(object_sample_reference, object_sample_query,
                                                             object_type=object_type)
        query_model_remapped = [apply_rotor(l, rotor).normal() for l in object_sample_query]
        new_cost = object_set_cost_sum(reference_model, query_model_remapped, object_type=object_type)
        if new_cost < min_cost:
            min_cost = new_cost
            min_rotor = rotor
        print('SAMPLE: ', i, '  cost  ', min_cost)
    return min_rotor, min_cost



def estimate_rotor_lines(reference_model, query_model, maxfev=20000, print_res=False):
    return estimate_rotor_objects(reference_model, query_model, maxfev=maxfev, print_res=print_res, object_type='lines')


def estimate_rotor_objects(reference_model, query_model, maxfev=20000, print_res=False, object_type='generic'):
    """
    Estimates the rotor that takes one set of objects to another
    """
    x0 = np.finfo(float).eps * np.random.rand(6)

    def minimisation_func(x):
        R = rotorconversion(x)
        query_model_remapped = [(apply_rotor(l, R)).normal() for l in query_model]
        return object_set_cost_sum(reference_model, query_model_remapped, object_type=object_type)

    res = minimize(minimisation_func, x0, method='SLSQP', options={'ftol': 10.0 ** (-16), \
                                                                      'maxiter': 500, \
                                                                      'disp': False})
    if print_res:
        print(res)
    res = minimize(minimisation_func, res.x, method='L-BFGS-B', options={'ftol':10.0**(-16), \
                                                                       'maxiter':500, \
                                                                       'disp': False, \
                                                                        'maxls':40})
    if print_res:
        print(res)
    rotor = rotorconversion(res.x)
    query_model_remapped = [apply_rotor(l, rotor).normal() for l in query_model]
    cost = object_set_cost_sum(reference_model, query_model_remapped, object_type=object_type)
    return rotor, cost


def cartans_lines(obj_list_a, obj_list_b):
    """ Performs the extended cartans algorithm as suggested by Alex Arsenovic """
    V_found,rs = cartan(A=obj_list_a,B=obj_list_b)
    theta = ((V_found*~V_found)*e1234)(0)
    V_found = e**(-theta/2*e123inf)*V_found
    return V_found


def sequential_object_rotor_estimation(reference_model, query_model, n_iterations=500,
                                       cost_tolerance=10*(10**-16), random_sequence=False,
                                       object_type='generic'):
    """
    Performs a sequential rotor update based on the rotors between individual objects
    Exits when the sum of the cost of rotor updates through the list is very small
    """
    R_total = 1.0 + 0.0*e1
    r_list = []
    for j in range(n_iterations):
        cost_sum = 0.0
        if random_sequence:
            indices = random.sample(range(len(query_model)), len(query_model))
        else:
            indices = range(len(query_model))
        for i in indices:
            C1 = (apply_rotor(query_model[i], R_total)).normal()
            C2 = reference_model[i]
            if abs(C1 + C2) < 0.0001:
                C1 = -C1
            if object_type == 'lines':
                rroot = (square_roots_of_rotor(rotor_between_lines(C1, C2))[0]).normal()
            else:
                rroot = (square_roots_of_rotor(rotor_between_objects(C1, C2))[0]).normal()
            r_list.append(rroot)
            R_total = (rroot*R_total).normal()
            cost_sum += rotor_cost(rroot)
        if cost_sum < cost_tolerance:
            exit_flag = 0
            return R_total, r_list, exit_flag
    exit_flag = 1
    return R_total, r_list, exit_flag


def sequential_object_rotor_estimation_convergence_detection(reference_model, query_model, n_iterations=500,
                                                             cost_tolerance=10*(10**-16), random_sequence=False,
                                                             object_type='generic'):
    """
    Performs a sequential rotor update based on the rotors between individual objects
    Exits when a full rotation through all objects produces a very small update of rotor
    """
    grade_list = [grade_obj(ob) for ob in query_model]

    R_total = 1.0 + 0.0*e1
    r_list = []
    for j in range(n_iterations):
        r_set = 1.0 + 0.0*e1
        if random_sequence:
            indices = random.sample(range(len(query_model)), len(query_model))
        else:
            indices = range(len(query_model))
        for i in indices:
            grade = grade_list[i]
            new_obj = apply_rotor(query_model[i],R_total)(grade)
            C1 = normalised(new_obj)
            C2 = reference_model[i]
            if abs(C1 + C2) < 0.0001:
                C1 = -C1
            if object_type == 'lines':
                rroot = normalised(square_roots_of_rotor((rotor_between_objects(C1,C2)))[0])
            else:
                rroot = normalised(square_roots_of_rotor((rotor_between_objects(C1,C2)))[0])
            r_list.append(rroot)
            r_set = normalised(rroot*r_set)
            R_total = normalised(rroot * R_total)
        if rotor_cost(r_set) < cost_tolerance:
            exit_flag = 0
            return R_total, r_list, exit_flag
    exit_flag = 1
    return R_total, r_list, exit_flag



