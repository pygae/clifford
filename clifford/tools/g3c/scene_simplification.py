
import numpy as np
from . import average_objects
from .cost_functions import object_set_cost_matrix, object_cost_function
from .cuda import object_set_cost_cuda_mvs
import random
from clifford.g3c import *


def average_symmetric(C1, C2):
    if object_cost_function(C1, C2) < object_cost_function(C1, -C2):
        return average_objects([C1, C2])
    else:
        return average_objects([C1, -C2])


def simplify_scene_recursive(objects, threshold):
    # Calculate the cost between this object and every other object
    cost_mat = object_set_cost_matrix(objects, objects)
    # Set the diagonals to threshold*2
    np.fill_diagonal(cost_mat, np.finfo(float).max)
    # Take the smallest value
    min_index = np.unravel_index(np.nanargmin(cost_mat), cost_mat.shape)
    min_value = cost_mat[min_index]
    # If the smallest value is above the value then return the current object set
    if min_value >= threshold:
        return objects
    # If it is less than the threshold then mean the two objects and recurse
    else:
        object_a = objects[min_index[0]]
        object_b = objects[min_index[1]]
        objects[min_index[1]] = average_objects([object_a, object_b])
        del objects[min_index[0]]
        return simplify_scene_recursive(objects, threshold)


def simplify_scene(objects, threshold, symmetric=False):
    o_copy = [o for o in objects]
    cost_matrix = object_set_cost_matrix(o_copy, o_copy, symmetric=symmetric)
    np.fill_diagonal(cost_matrix, np.finfo(float).max)
    while True:
        # Take the smallest value
        min_index = np.unravel_index(np.nanargmin(cost_matrix), cost_matrix.shape)
        min_value = cost_matrix[min_index]
        # If the smallest value is above the value then return the current object set
        if min_value >= threshold:
            return [o for o in objects if not (o == 0 * e1)]
        # If it is less than the threshold then mean the two objects and recurse
        else:
            a_ind = min_index[0]
            b_ind = min_index[1]
            object_a = objects[a_ind]
            object_b = objects[b_ind]
            if symmetric:
                objects[b_ind] = average_symmetric(object_a, object_b)
            else:
                objects[b_ind] = average_objects([object_a, object_b])
            objects[a_ind] = 0*e1
            cost_matrix[:, a_ind] = np.finfo(float).max
            cost_matrix[a_ind, :] = np.finfo(float).max
            cost_to_others = object_set_cost_matrix([objects[b_ind]], objects, symmetric=symmetric)
            cost_matrix[:, b_ind] = cost_to_others
            cost_matrix[b_ind, :] = cost_to_others
            np.fill_diagonal(cost_matrix, np.finfo(float).max)
