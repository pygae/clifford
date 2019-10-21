
import numpy as np
import random
import copy

from .cost_functions import object_set_cost_matrix, object_set_cost_matrix_sum
from . import average_objects
from .rotor_estimation import estimate_rotor_objects
from .GAOnline import GAScene

from .cuda import object_set_cost_cuda_mvs


def compare_labels(old_labels, new_labels):
    """
    Compares two lists of labels to see if any of them are different
    """
    for i in range(len(old_labels)):
        ol = old_labels[i]
        nl = new_labels[i]
        if ol != nl:
            return False
    return True


def assign_measurements_to_objects_matrix(objects, objects_measurements, object_type='generic',
                                          cuda=False, symmetric=False):
    """
    Assigns each object in objects_measurements to one in objects based on minimum cost
    """
    if cuda:
        matrix = object_set_cost_cuda_mvs(objects, objects_measurements)
    else:
        matrix = object_set_cost_matrix(objects, objects_measurements,
                                        object_type=object_type,
                                        symmetric=symmetric)
    labels = np.nanargmin(matrix, axis=0)
    costs = np.array([matrix[l, i] for i, l in enumerate(labels)])
    return [labels, costs]


proximity_match = assign_measurements_to_objects_matrix


def fit_object_to_objects(object_start, objects, averaging_method='unweighted'):
    """
    find the object at the center of the objects passed
    averaging_method flag determines if bivector parameterised optimisation or averaging of objects is used
    """
    if len(objects) > 0:
        if averaging_method == 'bivector':
            # print('Checking other starting possibilities')
            min_cost = object_set_cost_matrix_sum([object_start], objects)
            for a in objects:
                this_cost = object_set_cost_matrix_sum([a], objects)
                if this_cost < min_cost:
                    min_cost = this_cost
                    object_start = a
            # print('Beginning optimisation')
            rotor, cost = estimate_rotor_objects(objects, [object_start])
            final_object = (rotor * object_start * ~rotor).normal()
        elif averaging_method == 'unweighted':
            final_object = average_objects(objects)
        elif averaging_method == 'weighted':
            final_object = 1.0*object_start
            for i in range(10):
                weights = (1.0 / (object_set_cost_matrix([final_object], objects) + 0.0001)).flatten()
                final_object = average_objects(objects, weights=weights)
        else:
            raise ValueError('No averaging method defined')
    else:
        return 1.0*object_start
    return final_object


def reassign_unused_centroids(labels, costs, n):
    """ Reassigns unused centroids to the largest cost points """
    ind = np.argpartition(costs, -n)[-n:]
    max_cost_labels_index = ind[np.argsort(costs[ind])]
    k = 0
    new_labels = copy.deepcopy(labels)
    for i in range(n):
        if not (i in labels):
            new_labels[max_cost_labels_index[k]] = i
            k = k + 1
    return new_labels


def n_clusters_objects(n, objects_measurements, initial_centroids=None,
                       n_shotgunning=1, averaging_method='unweighted'):
    """
    Performs n means clustering with geometric objects
    Includes shotgunning for initial cluster position
    averaging_method flag determines if bivector parameterised optimisation or averaging of objects is used
    """
    # print("Initialising")
    min_shotgun_cost = np.finfo(float).max
    for i in range(n_shotgunning):
        if initial_centroids is None:
            # Randomly start the centroids
            if n <= len(objects_measurements):
                centroid_indices = random.sample(range(len(objects_measurements)), n)
            else:
                centroid_indices = random.sample(range(len(objects_measurements)), len(objects_measurements))
            centroids = [objects_measurements[i] for i in centroid_indices]
        else:
            if len(initial_centroids) < n:
                if len(objects_measurements) > n - len(initial_centroids):
                    centroids = initial_centroids + [objects_measurements[i] for i in
                                                     random.sample(range(len(objects_measurements)),
                                                                   n - len(initial_centroids))]
                else:
                    centroids = initial_centroids[:]
            else:
                centroids = initial_centroids[:]

        t_old_labels, t_old_costs = assign_measurements_to_objects_matrix(centroids, objects_measurements)
        t_shotgun_cost = sum(t_old_costs)
        if t_shotgun_cost < min_shotgun_cost:
            min_shotgun_cost = t_shotgun_cost
            old_labels = [l for l in t_old_labels]
            start_centroids = [c for c in centroids]
            start_labels = [l for l in old_labels]

    # print("Clustering")
    for i in range(10000):
        # Optimise the centroids to fit better their respective clusters
        new_centroids = []
        for centroid_index, object_start in enumerate(centroids):
            assigned_measurements = [object for i, object in enumerate(objects_measurements) if
                                     centroid_index == old_labels[i]]
            new_object = fit_object_to_objects(object_start, assigned_measurements, averaging_method=averaging_method)
            new_centroids.append(new_object)
        centroids = [c for c in new_centroids]
        # Assign the objects to the nearest centroid
        new_labels, new_costs = assign_measurements_to_objects_matrix(centroids, objects_measurements)

        # Reassign any unused centroids to the line with largest cost
        new_labels = reassign_unused_centroids(new_labels, new_costs, n)

        # If nothing has changed we have reached the optimum
        if compare_labels(old_labels, new_labels):
            return [new_labels, centroids, start_labels, start_centroids]
        else:
            old_labels = [copy.deepcopy(l) for l in new_labels]
    return [new_labels, centroids, start_labels, start_centroids]


def visualise_n_clusters(all_objects, centroids, labels, object_type='line',
                         color_1=np.array([255, 0, 0]), color_2=np.array([0, 255, 0])):
    """
    Utility method for visualising several clusters and their respective centroids
    using GAOnline
    """
    alpha_list = np.linspace(0, 1, num=len(centroids))
    sc = GAScene()
    for ind, this_obj in enumerate(all_objects):
        alpha = alpha_list[labels[ind]]
        cluster_color = (alpha * color_1 + (1 - alpha) * color_2)
        color_string = 'rgb' + str(tuple([int(c) for c in cluster_color]))
        sc.add_object(this_obj, object_type, color_string)

    for c in centroids:
        sc.add_object(c, object_type, 'rgb(0,0,0)')

    return sc
