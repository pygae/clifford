
from .object_clustering import assign_measurements_to_objects_matrix, compare_labels
from .rotor_estimation import estimate_rotor_objects, estimate_rotor_objects_subsample, \
    sequential_object_rotor_estimation, sequential_object_rotor_estimation_convergence_detection, \
    estimate_rotor_objects_subsample_sequential
from .rotor_parameterisation import interpolate_TR_rotors
from .cost_functions import val_rotor_cost_sparse
from . import apply_rotor
import numpy as np
from clifford.g3c import *
from clifford.tools.g3c.cuda import sequential_rotor_estimation_cuda_mvs

import clifford as cf


def newton_rotor(r, delta_r, delta_delta_r, max_steps=10):
    alpha = 1.0/((1.0/max_steps) + val_rotor_cost_sparse(delta_delta_r.normal().value)/val_rotor_cost_sparse(delta_r.normal().value))
    return interpolate_TR_rotors(delta_r*r, r, 1 + alpha)


def iterative_model_match_sequential(reference_model, query_model, iterations=100,
                          rotor_newton=False, max_newton_steps=10,
                          object_type='generic', cuda=False, print_rotor=False, r_track=None):
    if rotor_newton:
        # Set up the newton stuff
        delta_r_old = 1.0 + 0.0 * e1
        r_old = 1.0 + 0.0 * e1

    # Get the starting labels
    labels, costs = assign_measurements_to_objects_matrix(reference_model, query_model,
                                                          object_type=object_type, cuda=cuda)
    old_labels = [l for l in labels]
    remapped_objects = [o for o in query_model]
    r_est = 1.0 + 0.0*e1
    assert iterations > 0, 'Must have at least 1 iteration'
    for i in range(iterations):
        # Reorder
        reordered_list_ref = [reference_model[i] for i in labels]
        # Estimate the rotor
        r_est_update, r_list, exit_flag = sequential_object_rotor_estimation_convergence_detection(reordered_list_ref, remapped_objects,
                                                                             random_sequence=True, n_iterations=10,
                                                                             object_type=object_type)
        # Now update our running estimate
        r_est = (r_est_update*r_est)
        r_est = r_est.normal()
        # Re map with our new rotor
        remapped_objects = [apply_rotor(l, r_est).normal() for l in query_model]
        # Get the new matching
        labels, costs = assign_measurements_to_objects_matrix(reference_model, remapped_objects,
                                                              object_type=object_type, cuda=cuda)
        if r_track is not None:
            r_track.append(r_est)
        if print_rotor:
            print(r_est)
        print(i)
        if compare_labels(old_labels, labels):
            return labels, costs, r_est
        old_labels = [l for l in labels]
    return labels, costs, r_est


def iterative_model_match(reference_model, query_model, iterations=100,
                          rotor_newton=False, max_newton_steps=10,
                          object_type='generic', cuda=False):

    if rotor_newton:
        # Set up the newton stuff
        delta_r_old = 1.0 + 0.0 * e1
        r_old = 1.0 + 0.0 * e1

    # Get the starting labels
    labels, costs = assign_measurements_to_objects_matrix(reference_model, query_model,
                                                          object_type=object_type, cuda=cuda)
    old_labels = [+l for l in labels]
    remapped_objects = [o for o in query_model]
    r_est = 1.0 + 0.0*e1
    assert iterations > 0, 'Must have at least 1 iteration'
    for i in range(iterations):
        # Reorder
        reordered_list_ref = [reference_model[i] for i in labels]
        # Estimate the rotor
        r_est_update, cost = estimate_rotor_objects(reordered_list_ref, remapped_objects,
                                                    object_type=object_type)
        # Now update our running estimate
        r_est = (r_est_update*r_est)
        r_est = r_est.normal()

        if rotor_newton:
            # Do our newton style stuff
            delta_r = (r_est * (~r_old)).normal()
            delta_delta_r = delta_r * (~delta_r_old)
            delta_r_old = (r_est * (~r_old)).normal()
            r_est = newton_rotor(r_old, delta_r, delta_delta_r, max_newton_steps)
            r_old = 1.0*r_est
            r_est = r_est.normal()

        # Re map with our new rotor
        remapped_objects = [apply_rotor(l, r_est).normal() for l in query_model]
        # Get the new matching
        labels, costs = assign_measurements_to_objects_matrix(reference_model, remapped_objects,
                                                              object_type=object_type, cuda=cuda)
        if compare_labels(old_labels, labels):
            return labels, costs, r_est
        old_labels = [+l for l in labels]
        print(i)
    return labels, costs, r_est


def REFORM(reference_model, query_model, n_samples, objects_per_sample,
           iterations=100, covergence_threshold=0.00000001,
           pool_size=1,object_type='generic', cuda=False, print_rotor=False):
    # Get the starting labels
    labels, costs = assign_measurements_to_objects_matrix(reference_model, query_model,
                                                          object_type=object_type, cuda=cuda)

    min_global_cost = np.inf
    min_global_rotor = 1.0 + 0.0 * e1

    r_est = 1.0 + 0.0 * e1
    remapped_objects = [o for o in query_model]

    assert iterations > 0, 'Must have at least 1 iteration'
    for i in range(iterations):
        # Reorder and estimate the rotor
        reordered_list_a = [reference_model[i] for i in labels]
        r_est_update, cost = estimate_rotor_objects_subsample(reordered_list_a, remapped_objects,
                                                              n_samples,
                                                              objects_per_sample,
                                                              pool_size=pool_size,
                                                              object_type=object_type)
        r_est = (r_est_update * r_est)
        r_est = r_est.normal()
        # Re map with our new rotor
        remapped_objects = [apply_rotor(l,r_est).normal() for l in query_model]
        # Get the new matching
        labels, costs = assign_measurements_to_objects_matrix(reference_model, remapped_objects,
                                                              object_type=object_type, cuda=cuda)
        current_cost = np.sum(costs)
        if print_rotor:
            print(r_est)
        print(i, current_cost, covergence_threshold)
        if current_cost < min_global_cost:
            min_global_cost = current_cost
            min_global_rotor = +r_est
        if current_cost < covergence_threshold:
            return labels, costs, r_est
    # Re map with our new rotor
    remapped_objects = [apply_rotor(l, min_global_rotor).normal() for l in query_model]
    # Get the new matching
    labels, costs = assign_measurements_to_objects_matrix(reference_model, remapped_objects, cuda=True)
    return labels, costs, min_global_rotor


def REFORM_sequential(reference_model, query_model, n_samples, objects_per_sample,
                      iterations=100, covergence_threshold=0.00000001,
           pool_size=1,object_type='generic', cuda=False):
    # Get the starting labels
    labels, costs = assign_measurements_to_objects_matrix(reference_model, query_model,
                                                          object_type=object_type, cuda=cuda)

    min_global_cost = np.inf
    min_global_rotor = 1.0 + 0.0 * e1

    r_est = 1.0 + 0.0 * e1
    remapped_objects = [o for o in query_model]

    assert iterations > 0, 'Must have at least 1 iteration'
    for i in range(iterations):
        # Reorder and estimate the rotor
        reordered_list_a = [reference_model[i] for i in labels]
        r_est_update, cost = estimate_rotor_objects_subsample_sequential(reordered_list_a, remapped_objects,
                                                              n_samples,
                                                              objects_per_sample,
                                                              pool_size=pool_size,
                                                              object_type=object_type)
        r_est = (r_est_update * r_est)
        r_est = r_est.normal()
        # Re map with our new rotor
        remapped_objects = [apply_rotor(l,r_est).normal() for l in query_model]
        # Get the new matching
        labels, costs = assign_measurements_to_objects_matrix(reference_model, remapped_objects,
                                                              object_type=object_type, cuda=cuda)
        current_cost = np.sum(costs)
        print(i, current_cost, covergence_threshold)
        if current_cost < min_global_cost:
            min_global_cost = current_cost
            min_global_rotor = +r_est
        if current_cost < covergence_threshold:
            return labels, costs, r_est
    # Re map with our new rotor
    remapped_objects = [apply_rotor(l, min_global_rotor).normal() for l in query_model]
    # Get the new matching
    labels, costs = assign_measurements_to_objects_matrix(reference_model, remapped_objects, cuda=True)
    return labels, costs, min_global_rotor


def REFORM_cuda(reference_model, query_model, n_samples, objects_per_sample, iterations=100,
                covergence_threshold=0.00000001, mutation_probability=None):
    # Get the starting labels
    labels, costs = assign_measurements_to_objects_matrix(reference_model, query_model, cuda=True)

    min_global_cost = np.inf
    min_global_rotor = 1.0 + 0.0 * e1

    r_est = 1.0 + 0.0 * e1
    remapped_objects = [o for o in query_model]
    assert iterations > 0, 'Must have at least 1 iteration'
    for i in range(iterations):
        # Reorder and estimate the rotor
        reordered_list_a = [reference_model[i] for i in labels]
        r_list, cost_array = sequential_rotor_estimation_cuda_mvs(reordered_list_a,
                                                                       remapped_objects,
                                                              n_samples,
                                                              objects_per_sample,
                                                                  mutation_probability=mutation_probability)
        min_cost_index = np.argmin(cost_array)
        min_cost = cost_array[min_cost_index]
        r_est_update = r_list[min_cost_index]
        r_est = (r_est_update * r_est)
        r_est = r_est.normal()
        # Re map with our new rotor
        remapped_objects = [apply_rotor(l,r_est).normal() for l in query_model]
        # Get the new matching
        labels, costs = assign_measurements_to_objects_matrix(reference_model, remapped_objects, cuda=True)
        current_cost = np.sum(costs)
        print(i, covergence_threshold, current_cost, min_global_cost)
        if current_cost < min_global_cost:
            min_global_cost = current_cost
            min_global_rotor = +r_est
        if current_cost < covergence_threshold:
            return labels, costs, r_est
    # Re map with our new rotor
    remapped_objects = [apply_rotor(l, min_global_rotor).normal() for l in query_model]
    # Get the new matching
    labels, costs = assign_measurements_to_objects_matrix(reference_model, remapped_objects, cuda=True)
    return labels, costs, min_global_rotor