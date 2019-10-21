
from .cuda_products import gmt_func as gp_device
from .cuda_products import imt_func as ip_device
import numpy as np
import numba.cuda
import numba
import math
import random

from . import *


def sequential_rotor_estimation_chunks(reference_model_array, query_model_array, n_samples, n_objects_per_sample, mutation_probability=None):

    # Stack up a list of numbers
    total_matches = n_samples*n_objects_per_sample
    sample_indices = random.sample(range(total_matches), total_matches)

    n_mvs = reference_model_array.shape[0]
    sample_indices = [i % n_mvs for i in sample_indices]

    if mutation_probability is not None:
        reference_model_array_new = []
        mutation_flag = np.random.binomial(1, mutation_probability, total_matches)
        for mut, i in zip(mutation_flag, sample_indices):
            if mut:
                ref_ind = random.sample(range(len(reference_model_array)), 1)[0]
            else:
                ref_ind = i
            reference_model_array_new.append(reference_model_array[ref_ind, :])
        reference_model_array_new = np.array(reference_model_array_new)
    else:
        reference_model_array_new = np.array([reference_model_array[i, :] for i in sample_indices], dtype=np.float64)
    query_model_array_new = np.array([query_model_array[i, :] for i in sample_indices], dtype=np.float64)

    output = np.zeros((n_samples, 32), dtype=np.float64)
    cost_array = np.zeros(n_samples, dtype=np.float64)

    sequential_rotor_estimation_chunks_jit(reference_model_array_new, query_model_array_new, output, cost_array)

    return output, cost_array


def sequential_rotor_estimation_chunks_mvs(reference_model_list, query_model_list, n_samples, n_objects_per_sample, mutation_probability=None):
    query_model_array = np.array([l.value for l in query_model_list])
    reference_model_array = np.array([l.value for l in reference_model_list])
    output, cost_array = sequential_rotor_estimation_chunks(reference_model_array, query_model_array, n_samples, n_objects_per_sample, mutation_probability=mutation_probability)
    output_mvs = [query_model_list[0]._newMV(output[i, :]) for i in range(output.shape[0])]
    return output_mvs, cost_array






























@numba.cuda.jit(device=True)
def set_as_unit_rotor_device(array):
    for j in range(1, 32):
        array[j] = 0.0
    array[0] = 1.0


@numba.cuda.jit(device=True)
def sequential_rotor_estimation_device(reference_model, query_model, rotor_output):
    n_iterations = 20
    cost_tolerance = 10 * (10 ** -16)

    # Allocate memory
    r_set = numba.cuda.local.array(32, dtype=numba.float64)
    r_running = numba.cuda.local.array(32, dtype=numba.float64)
    r_root = numba.cuda.local.array(32, dtype=numba.float64)
    r_temp = numba.cuda.local.array(32, dtype=numba.float64)
    C1 = numba.cuda.local.array(32, dtype=numba.float64)

    # Set up the running rotor estimate
    set_as_unit_rotor_device(r_running)

    # Start iterating for convergence
    for iteration_number in range(n_iterations):

        # Set up the convergence check
        set_as_unit_rotor_device(r_set)

        # Iterate over the array of objects
        for mv_ind in range(reference_model.shape[0]):
            apply_rotor_device(query_model[mv_ind, :], r_running, C1)
            normalise_mv_device(C1)
            C2 = reference_model[mv_ind, :]

            # Check if they are the same other than a sign flip
            sum_abs = 0.0
            for b_ind in range(32):
                sum_abs += abs(C1[b_ind] + C2[b_ind])
            if sum_abs < 0.0001:
                set_as_unit_rotor_device(r_root)
            else:
                rotor_between_objects_device(C1, C2, r_temp)
                square_root_of_rotor_device(r_temp, r_root)

            # Update the set rotor and the running rotor
            gp_device(r_root, r_set, r_temp)
            normalise_mv_copy_device(r_temp, r_set)
            gp_device(r_root, r_running, r_temp)
            normalise_mv_copy_device(r_temp, r_running)

            # Check if we have converged
            if rotor_cost_device(r_set) < cost_tolerance:
                normalise_mv_copy_device(r_running, rotor_output)

                # Now calculate the cost of this transform
                total_cost = 0.0
                for object_ind in range(query_model.shape[0]):
                    apply_rotor_device(query_model[object_ind, :], rotor_output, r_temp)
                    total_cost += cost_between_objects_device(r_temp, reference_model[object_ind, :])
                return total_cost
    # Return whatever we have
    normalise_mv_copy_device(r_running, rotor_output)

    total_cost = 0.0
    for object_ind in range(query_model.shape[0]):
        apply_rotor_device(query_model[object_ind, :], rotor_output, r_temp)
        total_cost += cost_between_objects_device(r_temp, reference_model[object_ind, :])
    return total_cost


@numba.cuda.jit
def sequential_rotor_estimation_kernel(reference_model, query_model, output, cost_array):
    # Break the model into n chunks and estimate the rotor based on each of those
    n_chunks = output.shape[0]
    n_objects_per_chunk = int(reference_model.shape[0]/n_chunks)
    i = numba.cuda.grid(1)
    if i < n_chunks:
        ref = reference_model[i*n_objects_per_chunk:(i+1)*n_objects_per_chunk]
        qer = query_model[i*n_objects_per_chunk:(i+1)*n_objects_per_chunk]
        total_cost = sequential_rotor_estimation_device(ref, qer, output[i, :])
        cost_array[i] = total_cost


def sequential_rotor_estimation_cuda(reference_model_array, query_model_array, n_samples=None, n_objects_per_sample=None, mutation_probability=None):

    if n_samples is None:
        n_samples = int(len(query_model_array)/2)
    if n_objects_per_sample is None:
        n_objects_per_sample = max(int(len(query_model_array)/10), 5)

    # Stack up a list of numbers
    total_matches = n_samples*n_objects_per_sample
    sample_indices = random.sample(range(total_matches), total_matches)

    n_mvs = reference_model_array.shape[0]
    sample_indices = [i % n_mvs for i in sample_indices]

    if mutation_probability is not None:
        reference_model_array_new = []
        mutation_flag = np.random.binomial(1, mutation_probability, total_matches)
        for mut, i in zip(mutation_flag, sample_indices):
            if mut:
                ref_ind = random.sample(range(len(reference_model_array)), 1)[0]
            else:
                ref_ind = i
            reference_model_array_new.append(reference_model_array[ref_ind, :])
        reference_model_array_new = np.array(reference_model_array_new)
    else:
        reference_model_array_new = np.array([reference_model_array[i, :] for i in sample_indices], dtype=np.float64)
    query_model_array_new = np.array([query_model_array[i, :] for i in sample_indices], dtype=np.float64)

    output = np.zeros((n_samples, 32), dtype=np.float64)
    cost_array = np.zeros(n_samples, dtype=np.float64)

    blockdim = 64
    griddim = int(math.ceil(reference_model_array_new.shape[0] / blockdim))

    sequential_rotor_estimation_kernel[griddim, blockdim](reference_model_array_new, query_model_array_new, output, cost_array)

    return output, cost_array


def sequential_rotor_estimation_cuda_mvs(reference_model_list, query_model_list, n_samples, n_objects_per_sample, mutation_probability=None):
    query_model_array = np.array([l.value for l in query_model_list])
    reference_model_array = np.array([l.value for l in reference_model_list])
    output, cost_array = sequential_rotor_estimation_cuda(reference_model_array, query_model_array, n_samples, n_objects_per_sample, mutation_probability=mutation_probability)
    output_mvs = [query_model_list[0]._newMV(output[i, :]) for i in range(output.shape[0])]
    return output_mvs, cost_array


@numba.cuda.jit(device=True)
def apply_rotor_device(mv, rotor, output):
    rotor_adjoint = numba.cuda.local.array(32, dtype=numba.float64)
    temp = numba.cuda.local.array(32, dtype=numba.float64)
    adjoint_device(rotor, rotor_adjoint)
    gp_device(mv, rotor_adjoint, temp)
    gp_device(rotor, temp, output)


@numba.cuda.jit
def apply_rotor_kernel(mv, rotor, output):
    # This does elementwise gp with the input arrays into the ouput array
    i = numba.cuda.grid(1)
    if i < mv.shape[0]:
        apply_rotor_device(mv[i, :], rotor[i, :], output[i, :])


@numba.cuda.jit(device=True)
def square_root_of_rotor_device(rotor, rotor_root):
    k_value = numba.cuda.local.array(32, dtype=numba.float64)
    sigma_val = numba.cuda.local.array(32, dtype=numba.float64)
    C_val = numba.cuda.local.array(32, dtype=numba.float64)
    for i in range(32):
        C_val[i] = rotor[i]
    C_val[0] += 1.0
    gp_mult_with_adjoint(C_val, sigma_val)
    positive_root_device(sigma_val, k_value)
    annhilate_k_device(k_value, C_val, rotor_root)


@numba.cuda.jit
def square_root_of_rotor_kernel(value, output):
    i = numba.cuda.grid(1)
    if i < value.shape[0]:
        square_root_of_rotor_device(value[i, :], output[i, :])


@numba.cuda.jit(device=True)
def adjoint_device(value, output):
    for j in range(0, 6):
        output[j] = value[j]
    output[6] = -value[6]
    output[7] = -value[7]
    output[8] = -value[8]
    output[9] = -value[9]
    output[10] = -value[10]
    output[11] = -value[11]
    output[12] = -value[12]
    output[13] = -value[13]
    output[14] = -value[14]
    output[15] = -value[15]
    output[16] = -value[16]
    output[17] = -value[17]
    output[18] = -value[18]
    output[19] = -value[19]
    output[20] = -value[20]
    output[21] = -value[21]
    output[22] = -value[22]
    output[23] = -value[23]
    output[24] = -value[24]
    output[25] = -value[25]
    for j in range(26, 32):
        output[j] = value[j]


@numba.cuda.jit
def gp_kernel(value, other_value, output):
    # This does elementwise gp with the input arrays into the ouput array
    i = numba.cuda.grid(1)
    if i < value.shape[0]:
        gp_device(value[i, :], other_value[i, :], output[i, :])


@numba.cuda.jit
def adjoint_kernel(value, output):
    i = numba.cuda.grid(1)
    if i < value.shape[0]:
        adjoint_device(value[i, :], output[i, :])


@numba.cuda.jit
def ip_kernel(value, other_value, output):
    i = numba.cuda.grid(1)
    if i < value.shape[0]:
        ip_device(value[i, :], other_value[i, :], output[i, :])


@numba.cuda.jit(device=True)
def project_val_cuda(val, output, grade):
    for i in range(32):
        output[i] = 0.0
    if grade == 0:
        output[0] = val[0]
    elif grade == 1:
        for j in range(1, 6):
            output[j] = val[j]
    elif grade == 2:
        for j in range(6, 16):
            output[j] = val[j]
    elif grade == 3:
        for j in range(16, 26):
            output[j] = val[j]
    elif grade == 4:
        for j in range(26, 31):
            output[j] = val[j]
    elif grade == 5:
        output[31] = val[31]


@numba.njit
def calc_norm_device(mv_val):
    adj_value = numba.cuda.local.array(32, dtype=numba.float64)
    output_value = numba.cuda.local.array(32, dtype=numba.float64)
    adjoint_device(mv_val, adj_value)
    gp_device(adj_value, mv_val, output_value)
    return math.sqrt(abs(output_value[0]))


@numba.njit(device=True)
def normalise_mv_device(mv_val):
    norm = calc_norm_device(mv_val)
    for i in range(32):
        mv_val[i] = mv_val[i]/norm


@numba.njit(device=True)
def normalise_mv_copy_device(mv_val, copy_array):
    norm = calc_norm_device(mv_val)
    for i in range(32):
        copy_array[i] = mv_val[i]/norm


@numba.cuda.jit
def normalise_mvs_kernel(value_array):
    i = numba.cuda.grid(1)
    if i < value_array.shape[0]:
        normalise_mv_device(value_array[i, :])


@numba.cuda.jit(device=True)
def annhilate_k_device(K_val, C_val, output):
    k_4 = numba.cuda.local.array(32, dtype=numba.float64)
    project_val_cuda(K_val, k_4, 4)
    for i in range(32):
        k_4[i] = -k_4[i]
    k_4[0] += K_val[0]
    gp_device(k_4, C_val, output)
    normalise_mv_device(output)


@numba.jit(device=True)
def dorst_norm_val_device(sigma_val):
    """ Square Root of Rotors - Implements the norm of a rotor"""
    s_4 = numba.cuda.local.array(32, dtype=numba.float64)
    s_4_sqrd = numba.cuda.local.array(32, dtype=numba.float64)
    project_val_cuda(sigma_val, s_4, 4)
    gp_device(s_4, s_4, s_4_sqrd)
    sqrd_ans = sigma_val[0]*sigma_val[0] - s_4_sqrd[0]
    return math.sqrt(abs(sqrd_ans))


@numba.cuda.jit
def dorst_norm_val_kernel(value, output):
    i = numba.cuda.grid(1)
    if i < value.shape[0]:
        output[i] = dorst_norm_val_device(value[i, :])


@numba.cuda.jit(device=True)
def positive_root_device(sigma_val, result):
    """
    Square Root of Rotors - Evaluates the positive root
    """
    norm_s = dorst_norm_val_device(sigma_val)
    denominator = (math.sqrt(2.0*sigma_val[0] + 2.0*norm_s))
    for i in range(32):
        result[i] = sigma_val[i]/denominator
    result[0] = result[0] + norm_s/denominator


@numba.cuda.jit(device=True)
def rotor_between_objects_device(L1, L2, rotor):
    L1sqrd_val = numba.cuda.local.array(32, dtype=numba.float64)
    gp_device(L1, L1, L1sqrd_val)
    if L1sqrd_val[0] > 0:
        C_val = numba.cuda.local.array(32, dtype=numba.float64)
        sigma_val = numba.cuda.local.array(32, dtype=numba.float64)
        k_value = numba.cuda.local.array(32, dtype=numba.float64)
        gp_device(L2, L1, C_val)
        C_val[0] += 1.0
        gp_mult_with_adjoint(C_val, sigma_val)
        positive_root_device(sigma_val, k_value)
        annhilate_k_device(k_value, C_val, rotor)
    else:
        L21 = numba.cuda.local.array(32, dtype=numba.float64)
        L12 = numba.cuda.local.array(32, dtype=numba.float64)
        gp_device(L2, L1, L21)
        gp_device(L1, L2, L12)
        sumval = 0.0
        for i in range(32):
            if i == 0:
                sumval += abs(L12[i] + L21[i] - 2.0)
            else:
                sumval += abs(L12[i] + L21[i])
            rotor[i] = -L21[i]
        if sumval < 0.0000001:
            rotor[0] = rotor[0] - 1.0
        else:
            rotor[0] = rotor[0] + 1.0
        normalise_mv_device(rotor)


@numba.cuda.jit
def rotor_between_objects_kernel(value, other_value, output):
    i = numba.cuda.grid(1)
    if i < value.shape[0]:
        rotor_between_objects_device(value[i, :], other_value[i, :], output[i, :])


@numba.cuda.jit(device=True)
def cost_between_objects_device(L1, L2):
    R_val = numba.cuda.local.array(32, dtype=numba.float64)
    rotor_between_objects_device(L1, L2, R_val)
    return rotor_cost_device(R_val)


@numba.cuda.jit
def cost_between_objects_kernel(value, other_value, output):
    # This does elementwise gp with the input arrays into the output array
    i = numba.cuda.grid(1)
    if i < value.shape[0]:
        output[i] = cost_between_objects_device(value[i, :], other_value[i, :])


@numba.cuda.jit
def object_set_cost_kernel(line_set_a, line_set_b, cost_matrix):
    a_ind, b_ind = numba.cuda.grid(2)
    if a_ind < line_set_a.shape[0]:
        if b_ind < line_set_b.shape[0]:
            cost_matrix[a_ind, b_ind] = cost_between_objects_device(line_set_a[a_ind, :], line_set_b[b_ind, :])


def object_set_cost_cuda_value(line_set_a, line_set_b):
    threadsperblock = (16, 16)
    blockspergrid_x = math.ceil(line_set_a.shape[0] / threadsperblock[0])
    blockspergrid_y = math.ceil(line_set_b.shape[0] / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    cost_matrix = np.zeros((line_set_a.shape[0], line_set_b.shape[0]))
    object_set_cost_kernel[blockspergrid, threadsperblock](line_set_a, line_set_b, cost_matrix)
    return cost_matrix


def object_set_cost_cuda_mvs(line_set_a, line_set_b):
    line_set_a_vals = np.array([l.value for l in line_set_a])
    line_set_b_vals = np.array([l.value for l in line_set_b])
    return object_set_cost_cuda_value(line_set_a_vals, line_set_b_vals)


@numba.cuda.jit(device=True)
def rotor_between_lines_device(L1, L2, rotor):
    L21_val = numba.cuda.local.array(32, dtype=numba.float64)
    L12_val = numba.cuda.local.array(32, dtype=numba.float64)

    gp_device(L2, L1, L21_val)
    gp_device(L1, L2, L12_val)

    beta_val = numba.cuda.local.array(32, dtype=numba.float64)
    K_val = numba.cuda.local.array(32, dtype=numba.float64)
    for i in range(32):
        K_val[i] = L21_val[i] + L12_val[i]
        beta_val[i] = 0.0
    K_val[0] += 2.0

    project_val_cuda(K_val, beta_val, 4)

    alpha = 2.0 * K_val[0]

    denominator = math.sqrt(alpha / 2)

    normalisation_val = numba.cuda.local.array(32, dtype=numba.float64)
    output_val = numba.cuda.local.array(32, dtype=numba.float64)
    for i in range(32):
        if i == 0:
            numerator_val = 1.0 - beta_val[i] / alpha
        else:
            numerator_val = -beta_val[i] / alpha
        normalisation_val[i] = numerator_val / denominator
        output_val[i] = L21_val[i]

    output_val[0] += 1
    gp_device(normalisation_val, output_val, rotor)


@numba.cuda.jit
def rotor_between_lines_kernel(value, other_value, output):
    i = numba.cuda.grid(1)
    if i < value.shape[0]:
        rotor_between_lines_device(value[i, :], other_value[i, :], output[i, :])


@numba.cuda.jit(device=True)
def gp_mult_with_adjoint_to_scalar(value):
    other_value = numba.cuda.local.array(32, dtype=numba.float64)
    adjoint_device(value, other_value)
    return value[0] * other_value[0] + value[3] * other_value[3] + value[4] * other_value[4] - value[5] * other_value[
        5] - value[6] * other_value[6] - value[7] * other_value[7] - value[8] * other_value[8] + value[9] * other_value[
               9] - value[10] * other_value[10] - value[11] * other_value[11] + value[12] * other_value[12] - value[
               13] * other_value[13] + value[14] * other_value[14] + value[15] * other_value[15] + value[2] * \
           other_value[2] - value[16] * other_value[16] + value[18] * other_value[18] - value[19] * other_value[19] + \
           value[20] * other_value[20] + value[21] * other_value[21] - value[22] * other_value[22] + value[23] * \
           other_value[23] + value[24] * other_value[24] + value[25] * other_value[25] + value[26] * other_value[26] - \
           value[27] * other_value[27] - value[28] * other_value[28] - value[29] * other_value[29] - value[30] * \
           other_value[30] - value[17] * other_value[17] + value[1] * other_value[1] - value[31] * other_value[31]


@numba.cuda.jit(device=True)
def gp_mult_with_adjoint(value, output):
    other_value = numba.cuda.local.array(32, dtype=numba.float64)
    adjoint_device(value, other_value)
    gp_device(value, other_value, output)


@numba.cuda.jit(device=True)
def rotor_cost_device(R_val):
    translation_val = numba.cuda.local.array(32, dtype=numba.float64)
    rotation_val = numba.cuda.local.array(32, dtype=numba.float64)
    ep_val = numba.cuda.local.array(32, dtype=numba.float64)
    for i in range(32):
        ep_val[i] = 0.0
    ep_val[4] = 1.0
    ip_device(R_val, ep_val, translation_val)
    for i in range(32):
        rotation_val[i] = R_val[i]
    rotation_val[0] -= 1
    a = abs(gp_mult_with_adjoint_to_scalar(rotation_val))
    b = abs(gp_mult_with_adjoint_to_scalar(translation_val))
    return a + b


@numba.cuda.jit(device=True)
def cost_line_to_line_device(L1, L2):
    R_val = numba.cuda.local.array(32, dtype=numba.float64)
    rotor_between_lines_device(L1, L2, R_val)
    return rotor_cost_device(R_val)


@numba.cuda.jit
def cost_line_to_line_kernel(value, other_value, output):
    # This does elementwise gp with the input arrays into the output array
    i = numba.cuda.grid(1)
    if i < value.shape[0]:
        output[i] = cost_line_to_line_device(value[i, :], other_value[i, :])


@numba.cuda.jit
def line_set_cost_kernel(line_set_a, line_set_b, cost_matrix):
    a_ind, b_ind = numba.cuda.grid(2)
    if a_ind < line_set_a.shape[0]:
        if b_ind < line_set_b.shape[0]:
            cost_matrix[a_ind, b_ind] = cost_line_to_line_device(line_set_a[a_ind, :], line_set_b[b_ind, :])


def line_set_cost_cuda_value(line_set_a, line_set_b):
    threadsperblock = (16, 16)
    blockspergrid_x = math.ceil(line_set_a.shape[0] / threadsperblock[0])
    blockspergrid_y = math.ceil(line_set_b.shape[0] / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    cost_matrix = np.zeros((line_set_a.shape[0], line_set_b.shape[0]))
    line_set_cost_kernel[blockspergrid, threadsperblock](line_set_a, line_set_b, cost_matrix)
    return cost_matrix


def line_set_cost_cuda_mvs(line_set_a, line_set_b):
    line_set_a_vals = np.array([l.value for l in line_set_a])
    line_set_b_vals = np.array([l.value for l in line_set_b])
    return line_set_cost_cuda_value(line_set_a_vals, line_set_b_vals)
