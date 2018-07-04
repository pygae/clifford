
import numpy as np
from . import average_objects
from .cost_functions import object_set_cost_matrix


def simplify_scene(objects, threshold):
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
        return simplify_scene(objects, threshold)
