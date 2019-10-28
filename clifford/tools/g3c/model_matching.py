
from .object_clustering import assign_measurements_to_objects_matrix, compare_labels
from .rotor_estimation import estimate_rotor_objects, estimate_rotor_objects_subsample, \
    sequential_object_rotor_estimation, \
    estimate_rotor_objects_subsample_sequential
from .rotor_parameterisation import interpolate_TR_rotors
from .cost_functions import val_rotor_cost_sparse
from . import apply_rotor
from .cost_functions import object_set_cost_matrix, object_cost_function, check_p_cost
import numpy as np
from clifford.g3c import *
from clifford.tools.g3c.cuda import sequential_rotor_estimation_cuda_mvs

import clifford as cf


def fingerprint_function(obj, other_objects, nbins=10, max_cost=100.0, cost_func=check_p_cost):
    """
    Bins the cost of the object to all the other objects to make a metric
    for the scene
    """
    cost_list = [cost_func(obj, o_obj) for o_obj in other_objects]
    counts, bins = np.histogram(cost_list, bins=nbins, range=(0.0, max_cost))
    return counts


def match_by_fingerprint(reference_model, query_model, nbins=None, max_cost=None, cost_func=check_p_cost):
    """
    Matches the objects in two scenes based on their fingerprint functions
    """
    if nbins is None:
        nbins = len(query_model)
    if max_cost is None:
        a = np.max(np.max(np.abs(np.array([[cost_func(obj, o_obj) for o_obj in query_model] for obj in query_model]))))
        b = np.max(np.max(np.abs(np.array([[cost_func(obj, o_obj) for o_obj in reference_model] for obj in reference_model]))))
        max_cost = max(a, b)
    labels = []
    min_costs = []
    for j in range(len(query_model)):
        query_bins = fingerprint_function(query_model[j], query_model,
                                          nbins=nbins, max_cost=max_cost, cost_func=cost_func)

        min_cost_match = np.inf
        min_cost_ind = 0
        for i in range(len(reference_model)):
            ref_plane_bins = fingerprint_function(reference_model[i], reference_model,
                                                  nbins=nbins, max_cost=max_cost, cost_func=cost_func)
            cost_match = np.sum(np.abs(query_bins-ref_plane_bins))
            if cost_match < min_cost_match:
                min_cost_match = cost_match
                min_cost_ind = i
        labels.append(min_cost_ind)
        min_costs.append(min_cost_match)
    return labels, min_costs


def iterative_model_match_sequential(reference_model, query_model, iterations=100,
                                     object_type='generic', cuda=False, print_rotor=False,
                                     r_track=None, start_labels=None):
    """
    Matches the query model to the reference model and estimates the motor between them
    Assumes that every query model item has a corresponding reference model item, multiple
    query model items can match the same reference model item. Uses the sequential rotor estimation
    """

    # Get the starting labels
    if start_labels is None:
        labels, costs = assign_measurements_to_objects_matrix(reference_model, query_model,
                                                              object_type=object_type, cuda=cuda)
    else:
        labels = [+l for l in start_labels]
    old_labels = [l for l in labels]
    remapped_objects = [o for o in query_model]
    r_est = 1.0 + 0.0*e1
    assert iterations > 0, 'Must have at least 1 iteration'
    for i in range(iterations):
        # Reorder
        reordered_list_ref = [reference_model[i] for i in labels]
        # Estimate the rotor
        r_est_update, exit_flag = sequential_object_rotor_estimation(reordered_list_ref, remapped_objects,
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
                          object_type='generic', cuda=False, start_labels=None,
                          symmetric=False):
    """
    Matches the query model to the reference model and estimates the motor between them
    Assumes that every query model item has a corresponding reference model item, multiple
    query model items can match the same reference model item
    """

    # Get the starting labels
    if start_labels is None:
        labels, costs = assign_measurements_to_objects_matrix(reference_model, query_model,
                                                              object_type=object_type, cuda=cuda,
                                                              symmetric=symmetric)
    else:
        labels = [+l for l in start_labels]
    old_labels = [+l for l in labels]
    remapped_objects = [o for o in query_model]
    r_est = 1.0 + 0.0*e1
    assert iterations > 0, 'Must have at least 1 iteration'
    for i in range(iterations):
        # Reorder
        reordered_list_ref = [reference_model[i] for i in labels]
        # Estimate the rotor
        r_est_update, cost = estimate_rotor_objects(reordered_list_ref, remapped_objects,
                                                    object_type=object_type,
                                                    symmetric=symmetric)
        # Now update our running estimate
        r_est = (r_est_update*r_est)
        r_est = r_est.normal()

        # Re map with our new rotor
        remapped_objects = [apply_rotor(l, r_est).normal() for l in query_model]
        # Get the new matching
        labels, costs = assign_measurements_to_objects_matrix(reference_model, remapped_objects,
                                                              object_type=object_type, cuda=cuda,
                                                              symmetric=symmetric)
        if compare_labels(old_labels, labels):
            return labels, costs, r_est
        old_labels = [+l for l in labels]
        print(i)
    return labels, costs, r_est


def REFORM(reference_model, query_model, n_samples=100, objects_per_sample=5,
           iterations=100, covergence_threshold=0.00000001,
           pool_size=1, object_type='generic', cuda=False,
           print_rotor=False, start_labels=None, motor=True):
    #  Get the starting labels
    if start_labels is None:
        labels, costs = assign_measurements_to_objects_matrix(reference_model, query_model,
                                                              object_type=object_type, cuda=cuda)
    else:
        labels = [+l for l in start_labels]

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
                                                              object_type=object_type,
                                                              motor=motor)
        r_est = (r_est_update * r_est)
        r_est = r_est.normal()
        # Re map with our new rotor
        remapped_objects = [apply_rotor(l, r_est).normal() for l in query_model]
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
    labels, costs = assign_measurements_to_objects_matrix(reference_model, remapped_objects, cuda=cuda)
    return labels, costs, min_global_rotor


def REFORM_sequential(reference_model, query_model, n_samples=100, objects_per_sample=5,
                      iterations=100, covergence_threshold=0.00000001,
                      pool_size=1, object_type='generic', cuda=False, start_labels=None):

    #  Get the starting labels
    if start_labels is None:
        labels, costs = assign_measurements_to_objects_matrix(reference_model, query_model,
                                                              cuda=cuda)
    else:
        labels = [+l for l in start_labels]
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
        remapped_objects = [apply_rotor(l, r_est).normal() for l in query_model]
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

    print('Finished iterations')
    # Re map with our new rotor
    remapped_objects = [apply_rotor(l, min_global_rotor).normal() for l in query_model]
    # Get the new matching
    print('Rematching')
    labels, costs = assign_measurements_to_objects_matrix(reference_model, remapped_objects, cuda=cuda)
    print('REFORM complete')
    return labels, costs, min_global_rotor


def REFORM_cuda(reference_model, query_model, n_samples=100, objects_per_sample=5, iterations=100,
                covergence_threshold=0.00000001, mutation_probability=None, start_labels=None):
    #  Get the starting labels
    if start_labels is None:
        labels, costs = assign_measurements_to_objects_matrix(reference_model, query_model,
                                                              cuda=True)
    else:
        labels = [+l for l in start_labels]
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
        remapped_objects = [apply_rotor(l, r_est).normal() for l in query_model]
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
