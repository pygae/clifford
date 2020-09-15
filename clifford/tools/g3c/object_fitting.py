"""
Tools for fitting geometric primitives to point clouds

Object Fitting
==========================================================

.. autosummary::
    :toctree: generated/

    fit_circle
    val_fit_circle
    fit_line
    val_fit_line
    fit_sphere
    val_fit_sphere
    fit_plane
    val_fit_plane
"""

from . import *


@numba.njit
def val_fit_circle(point_list):
    """
    Performs Leo Dorsts circle fitting technique
    """
    # Check if there are just 3 points
    if point_list.shape[0] == 3:
        best_obj = point_list[0, :]
        for i in range(1, 3):
            best_obj = omt_func(best_obj, point_list[i, :])
        return val_normalised(best_obj)
    # Loop over our points and construct the matrix
    accumulator_matrix = np.zeros((32, 32))
    for i in range(point_list.shape[0]):
        # Get the point as a left gmt matrix
        P_i_l = get_left_gmt_matrix(point_list[i, :])
        # Multiply and add
        accumulator_matrix += P_i_l @ mask0 @ P_i_l
    accumulator_matrix = accumulator_matrix @ mask1
    # Find the eigenvalues of the matrix
    e_vals, e_vecs = np.linalg.eig(accumulator_matrix)
    # Find the smallest and second smallest non negative eigenvalues
    min_eval = np.inf
    min_eval2 = np.inf
    min_eval_index = -1
    min_eval_index2 = -1
    for i in range(len(e_vals)):
        this_e_val = e_vals[i]
        if this_e_val > 0:
            if this_e_val < min_eval:
                min_eval2 = min_eval
                min_eval = this_e_val
                min_eval_index2 = min_eval_index
                min_eval_index = i
            elif this_e_val < min_eval2:
                min_eval2 = this_e_val
                min_eval_index2 = i
    best_sphere = val_normalised(mask1@np.real(e_vecs[:, min_eval_index]))
    second_best_sphere = val_normalised(mask1@np.real(e_vecs[:, min_eval_index2]))
    best_circle = val_normalised(mask3@dual_func(omt_func(best_sphere, second_best_sphere)))
    return best_circle


def fit_circle(point_list):
    """
    Performs Leo Dorsts circle fitting technique
    """
    return layout.MultiVector(val_fit_circle(np.array([p.value for p in point_list])))


@numba.njit
def val_fit_line(point_list):
    """
    Does line fitting with combo J.Lasenbys method and L. Dorsts
    """
    # Check if there are just 2 points
    if point_list.shape[0] == 2:
        best_obj = point_list[0, :]
        for i in range(1, 2):
            best_obj = omt_func(best_obj, point_list[i, :])
        return val_normalised(omt_func(best_obj, ninf.value))
    accumulator_matrix = np.zeros((32, 32))
    for i in range(point_list.shape[0]):
        P_i_l = get_left_gmt_matrix(point_list[i, :])
        P_i_r = get_right_gmt_matrix(point_list[i, :])
        accumulator_matrix += mask3@P_i_l@P_i_r
    # Find the eigenvalues of the matrix
    e_vals, e_vecs = np.linalg.eig(accumulator_matrix)
    # Find the smallest non negative eigenvalue
    min_eval = np.inf
    min_eval_index = -1
    for i in range(len(e_vals)):
        if e_vals[i] < min_eval and e_vals[i] > 0:
            min_eval = e_vals[i]
            min_eval_index = i
    best_line = mask3@omt_func(dual_func(e_vecs[:, min_eval_index]), ninf.value)
    return val_normalised(best_line)


def fit_line(point_list):
    """
    Does line fitting with combo J.Lasenbys method and L. Dorsts
    """
    return layout.MultiVector(val_fit_line(np.array([p.value for p in point_list])))


@numba.njit
def val_fit_sphere(point_list):
    """
    Performs Leo Dorsts sphere fitting technique
    """
    # Check if there are just 4 points
    if point_list.shape[0] == 4:
        best_sphere = point_list[0, :]
        for i in range(1, 4):
            best_sphere = omt_func(best_sphere, point_list[i, :])
        return val_normalised(best_sphere)
    # Loop over our points and construct the matrix
    accumulator_matrix = np.zeros((32, 32))
    for i in range(point_list.shape[0]):
        # Get the point as a left gmt matrix
        P_i_l = get_left_gmt_matrix(point_list[i, :])
        # Multiply and add
        accumulator_matrix += P_i_l @ mask0 @ P_i_l
    accumulator_matrix = accumulator_matrix @ mask1
    # Find the eigenvalues of the matrix
    e_vals, e_vecs = np.linalg.eig(accumulator_matrix)
    # Find the smallest non negative eigenvalues
    min_eval = np.inf
    min_eval_index = -1
    for i in range(len(e_vals)):
        if e_vals[i] < min_eval and e_vals[i] > 0:
            min_eval = e_vals[i]
            min_eval_index = i
    best_sphere = val_normalised(mask4@dual_func(np.real(e_vecs[:, min_eval_index])))
    return best_sphere


def fit_sphere(point_list):
    """
    Performs Leo Dorsts sphere fitting technique
    """
    return layout.MultiVector(val_fit_sphere(np.array([p.value for p in point_list])))


@numba.njit
def val_fit_plane(point_list):
    """
    Does plane fitting with combo J.Lasenbys method and L. Dorsts
    """
    # Check if there are just 3 points
    if point_list.shape[0] == 3:
        best_obj = point_list[0, :]
        for i in range(1, 3):
            best_obj = omt_func(best_obj, point_list[i, :])
        return val_normalised(omt_func(best_obj, ninf.value))
    accumulator_matrix = np.zeros((32, 32))
    for i in range(point_list.shape[0]):
        P_i_l = get_left_gmt_matrix(point_list[i, :])
        P_i_r = get_right_gmt_matrix(point_list[i, :])
        accumulator_matrix += mask4@P_i_l@P_i_r
    e_vals, e_vecs = np.linalg.eig(accumulator_matrix)
    min_eval = np.inf
    min_eval_index = -1
    for i in range(len(e_vals)):
        if e_vals[i] < min_eval and e_vals[i] > 0:
            min_eval = e_vals[i]
            min_eval_index = i
    best_plane = val_normalised(mask4@e_vecs[:, min_eval_index])
    return best_plane


def fit_plane(point_list):
    """
    Does plane fitting with combo J.Lasenbys method and L. Dorsts
    """
    return layout.MultiVector(val_fit_plane(np.array([p.value for p in point_list])))
