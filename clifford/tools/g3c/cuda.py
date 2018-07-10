

import numpy as np
import numba
import math


@numba.cuda.jit(device=True)
def adjoint_device(value, output):
    for j in range(0,6):
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
    for j in range(26,32):
        output[j] = value[j]


@numba.cuda.jit
def gp_kernel(value, other_value, output):
    # This does elementwise gp with the input arrays into the ouput array
    i = numba.cuda.grid(1)
    if i < value.shape[0]:
        gp_device(value[i, :], other_value[i, :], output[i, :])


@numba.cuda.jit
def adjoint_kernel(value, output):
    # This does elementwise gp with the input arrays into the ouput array
    i = numba.cuda.grid(1)
    if i < value.shape[0]:
        adjoint_device(value[i, :], output[i, :])


@numba.cuda.jit
def ip_kernel(value, other_value, output):
    # This does elementwise gp with the input arrays into the ouput array
    i = numba.cuda.grid(1)
    if i < value.shape[0]:
        ip_device(value[i, :], other_value[i, :], output[i, :])


@numba.cuda.jit(device=True)
def project_val_cuda(val, output, grade):
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


@numba.cuda.jit(device=True)
def rotor_line_to_line_device(L1, L2, rotor):
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
def rotor_line_to_line_kernel(value, other_value, output):
    # This does elementwise gp with the input arrays into the ouput array
    i = numba.cuda.grid(1)
    if i < value.shape[0]:
        rotor_line_to_line_device(value[i, :], other_value[i, :], output[i, :])


@numba.cuda.jit(device=True)
def gp_mult_with_adjoint(value):
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
    a = abs(gp_mult_with_adjoint(rotation_val))
    b = abs(gp_mult_with_adjoint(translation_val))
    return a + b


@numba.cuda.jit(device=True)
def cost_line_to_line_device(L1, L2):
    R_val = numba.cuda.local.array(32, dtype=numba.float64)
    rotor_line_to_line_device(L1, L2, R_val)
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


def line_set_cost_cuda(line_set_a, line_set_b):
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
    threadsperblock = (16, 16)
    blockspergrid_x = math.ceil(line_set_a_vals.shape[0] / threadsperblock[0])
    blockspergrid_y = math.ceil(line_set_b_vals.shape[0] / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    cost_matrix = np.zeros((line_set_a_vals.shape[0], line_set_b_vals.shape[0]))
    line_set_cost_kernel[blockspergrid, threadsperblock](line_set_a_vals, line_set_b_vals, cost_matrix)
    return cost_matrix


@numba.cuda.jit(device=True)
def ip_device(value, other_value, output):
    output[0] = value[1] * other_value[1] - value[27] * other_value[27] + value[26] * other_value[26] + value[4] * \
                other_value[4] + value[25] * other_value[25] + value[24] * other_value[24] - value[5] * other_value[5] + \
                value[23] * other_value[23] - value[6] * other_value[6] - value[22] * other_value[22] - value[7] * \
                other_value[7] + value[21] * other_value[21] - value[8] * other_value[8] + value[20] * other_value[20] + \
                value[9] * other_value[9] - value[19] * other_value[19] - value[10] * other_value[10] + value[18] * \
                other_value[18] - value[11] * other_value[11] - value[17] * other_value[17] + value[12] * other_value[
                    12] - value[13] * other_value[13] - value[16] * other_value[16] + value[14] * other_value[14] + \
                value[3] * other_value[3] - value[28] * other_value[28] + value[15] * other_value[15] - value[31] * \
                other_value[31] - value[29] * other_value[29] + value[2] * other_value[2] - value[30] * other_value[30]
    output[1] = -value[11] * other_value[17] - value[4] * other_value[8] - value[10] * other_value[16] - value[16] * \
                other_value[10] + value[15] * other_value[21] - value[25] * other_value[29] + value[5] * other_value[
                    9] + value[12] * other_value[18] - value[17] * other_value[11] + value[22] * other_value[26] - \
                value[24] * other_value[28] - value[2] * other_value[6] - value[23] * other_value[27] - value[31] * \
                other_value[30] + value[6] * other_value[2] - value[13] * other_value[19] + value[18] * other_value[
                    12] - value[30] * other_value[31] + value[7] * other_value[3] + value[27] * other_value[23] - value[
                    19] * other_value[13] + value[8] * other_value[4] + value[20] * other_value[14] + value[21] * \
                other_value[15] + value[14] * other_value[20] - value[26] * other_value[22] - value[9] * other_value[
                    5] - value[3] * other_value[7] + value[29] * other_value[25] + value[28] * other_value[24]
    output[2] = value[23] * other_value[14] - value[18] * other_value[9] + value[21] * other_value[28] - value[6] * \
                other_value[1] - value[19] * other_value[26] + value[10] * other_value[3] + value[7] * other_value[16] - \
                value[22] * other_value[13] - value[9] * other_value[18] + value[30] * other_value[25] - value[13] * \
                other_value[22] - value[28] * other_value[21] - value[3] * other_value[10] + value[14] * other_value[
                    23] + value[1] * other_value[6] + value[15] * other_value[24] - value[27] * other_value[20] + value[
                    29] * other_value[31] + value[26] * other_value[19] + value[31] * other_value[29] - value[4] * \
                other_value[11] + value[16] * other_value[7] + value[20] * other_value[27] - value[25] * other_value[
                    30] - value[12] * other_value[5] + value[17] * other_value[8] + value[24] * other_value[15] + value[
                    11] * other_value[4] + value[5] * other_value[12] + value[8] * other_value[17]
    output[3] = -value[16] * other_value[6] - value[14] * other_value[5] - value[9] * other_value[20] + value[19] * \
                other_value[8] - value[10] * other_value[2] - value[18] * other_value[27] - value[12] * other_value[
                    23] + value[17] * other_value[26] + value[15] * other_value[25] + value[1] * other_value[7] + value[
                    13] * other_value[4] + value[11] * other_value[22] - value[31] * other_value[28] - value[6] * \
                other_value[16] + value[8] * other_value[19] + value[5] * other_value[14] + value[24] * other_value[
                    30] - value[23] * other_value[12] + value[25] * other_value[15] + value[22] * other_value[11] - \
                value[4] * other_value[13] + value[2] * other_value[10] + value[21] * other_value[29] - value[7] * \
                other_value[1] - value[26] * other_value[17] - value[29] * other_value[21] - value[30] * other_value[
                    24] + value[27] * other_value[18] - value[20] * other_value[9] - value[28] * other_value[31]
    output[4] = value[26] * other_value[16] - value[14] * other_value[25] - value[24] * other_value[12] + value[31] * \
                other_value[27] + value[3] * other_value[13] - value[16] * other_value[26] - value[13] * other_value[
                    3] - value[25] * other_value[14] + value[27] * other_value[31] + value[1] * other_value[8] + value[
                    29] * other_value[20] - value[15] * other_value[5] - value[12] * other_value[24] + value[2] * \
                other_value[11] - value[17] * other_value[6] - value[11] * other_value[2] + value[30] * other_value[
                    23] - value[19] * other_value[7] - value[20] * other_value[29] - value[9] * other_value[21] - value[
                    7] * other_value[19] - value[8] * other_value[1] - value[6] * other_value[17] - value[10] * \
                other_value[22] - value[18] * other_value[28] - value[21] * other_value[9] - value[22] * other_value[
                    10] + value[5] * other_value[15] - value[23] * other_value[30] + value[28] * other_value[18]
    output[5] = value[29] * other_value[19] - value[18] * other_value[6] - value[24] * other_value[11] + value[28] * \
                other_value[17] + value[31] * other_value[26] - value[16] * other_value[27] - value[20] * other_value[
                    7] + value[26] * other_value[31] - value[22] * other_value[30] + value[30] * other_value[22] + \
                value[27] * other_value[16] - value[23] * other_value[10] - value[19] * other_value[29] - value[25] * \
                other_value[13] - value[21] * other_value[8] - value[15] * other_value[4] - value[17] * other_value[
                    28] - value[6] * other_value[18] - value[9] * other_value[1] + value[1] * other_value[9] - value[
                    11] * other_value[24] + value[3] * other_value[14] - value[8] * other_value[21] - value[12] * \
                other_value[2] + value[2] * other_value[12] - value[7] * other_value[20] - value[13] * other_value[25] - \
                value[10] * other_value[23] - value[14] * other_value[3] + value[4] * other_value[15]
    output[6] = value[25] * other_value[31] - value[18] * other_value[5] + value[3] * other_value[16] - value[5] * \
                other_value[18] + value[17] * other_value[4] + value[14] * other_value[27] + value[27] * other_value[
                    14] - value[26] * other_value[13] + value[16] * other_value[3] + value[31] * other_value[25] - \
                value[13] * other_value[26] + value[15] * other_value[28] + value[4] * other_value[17] + value[28] * \
                other_value[15]
    output[7] = -value[2] * other_value[16] - value[24] * other_value[31] + value[26] * other_value[11] + value[19] * \
                other_value[4] + value[4] * other_value[19] + value[29] * other_value[15] - value[20] * other_value[5] - \
                value[5] * other_value[20] - value[31] * other_value[24] + value[15] * other_value[29] - value[27] * \
                other_value[12] - value[12] * other_value[27] - value[16] * other_value[2] + value[11] * other_value[26]
    output[8] = -value[10] * other_value[26] - value[28] * other_value[12] + value[23] * other_value[31] - value[21] * \
                other_value[5] - value[14] * other_value[29] - value[29] * other_value[14] - value[2] * other_value[
                    17] - value[19] * other_value[3] - value[12] * other_value[28] + value[31] * other_value[23] - \
                value[3] * other_value[19] - value[17] * other_value[2] - value[26] * other_value[10] - value[5] * \
                other_value[21]
    output[9] = -value[11] * other_value[28] + value[22] * other_value[31] - value[28] * other_value[11] - value[29] * \
                other_value[13] - value[4] * other_value[21] - value[13] * other_value[29] - value[21] * other_value[
                    4] - value[3] * other_value[20] - value[18] * other_value[2] - value[2] * other_value[18] - value[
                    10] * other_value[27] - value[20] * other_value[3] + value[31] * other_value[22] - value[27] * \
                other_value[10]
    output[10] = value[1] * other_value[16] - value[5] * other_value[23] + value[31] * other_value[21] - value[23] * \
                 other_value[5] + value[30] * other_value[15] + value[22] * other_value[4] - value[8] * other_value[
                     26] + value[16] * other_value[1] - value[26] * other_value[8] + value[4] * other_value[22] + value[
                     9] * other_value[27] + value[21] * other_value[31] + value[15] * other_value[30] + value[27] * \
                 other_value[9]
    output[11] = value[28] * other_value[9] + value[26] * other_value[7] - value[24] * other_value[5] - value[3] * \
                 other_value[22] - value[14] * other_value[30] + value[1] * other_value[17] - value[20] * other_value[
                     31] + value[7] * other_value[26] - value[22] * other_value[3] + value[9] * other_value[28] - value[
                     31] * other_value[20] + value[17] * other_value[1] - value[30] * other_value[14] - value[5] * \
                 other_value[24]
    output[12] = value[7] * other_value[27] + value[1] * other_value[18] - value[19] * other_value[31] + value[28] * \
                 other_value[8] - value[3] * other_value[23] - value[4] * other_value[24] - value[24] * other_value[4] - \
                 value[30] * other_value[13] - value[31] * other_value[19] + value[27] * other_value[7] - value[23] * \
                 other_value[3] - value[13] * other_value[30] + value[18] * other_value[1] + value[8] * other_value[28]
    output[13] = value[18] * other_value[31] + value[2] * other_value[22] - value[26] * other_value[6] + value[9] * \
                 other_value[29] + value[12] * other_value[30] + value[31] * other_value[18] + value[19] * other_value[
                     1] - value[5] * other_value[25] - value[25] * other_value[5] + value[1] * other_value[19] + value[
                     22] * other_value[2] - value[6] * other_value[26] + value[29] * other_value[9] + value[30] * \
                 other_value[12]
    output[14] = -value[25] * other_value[4] + value[31] * other_value[17] - value[27] * other_value[6] + value[23] * \
                 other_value[2] + value[29] * other_value[8] + value[11] * other_value[30] - value[4] * other_value[
                     25] + value[17] * other_value[31] + value[30] * other_value[11] + value[1] * other_value[20] + \
                 value[2] * other_value[23] - value[6] * other_value[27] + value[8] * other_value[29] + value[20] * \
                 other_value[1]
    output[15] = -value[30] * other_value[10] - value[29] * other_value[7] + value[1] * other_value[21] - value[28] * \
                 other_value[6] - value[31] * other_value[16] + value[2] * other_value[24] + value[25] * other_value[
                     3] + value[21] * other_value[1] - value[6] * other_value[28] - value[7] * other_value[29] - value[
                     10] * other_value[30] - value[16] * other_value[31] + value[24] * other_value[2] + value[3] * \
                 other_value[25]
    output[16] = value[31] * other_value[15] + value[15] * other_value[31] - value[27] * other_value[5] - value[4] * \
                 other_value[26] + value[5] * other_value[27] + value[26] * other_value[4]
    output[17] = -value[14] * other_value[31] + value[3] * other_value[26] + value[5] * other_value[28] - value[31] * \
                 other_value[14] - value[28] * other_value[5] - value[26] * other_value[3]
    output[18] = -value[31] * other_value[13] + value[3] * other_value[27] - value[28] * other_value[4] - value[13] * \
                 other_value[31] + value[4] * other_value[28] - value[27] * other_value[3]
    output[19] = -value[2] * other_value[26] + value[12] * other_value[31] + value[5] * other_value[29] + value[31] * \
                 other_value[12] + value[26] * other_value[2] - value[29] * other_value[5]
    output[20] = value[4] * other_value[29] + value[27] * other_value[2] + value[11] * other_value[31] + value[31] * \
                 other_value[11] - value[2] * other_value[27] - value[29] * other_value[4]
    output[21] = -value[10] * other_value[31] - value[31] * other_value[10] - value[2] * other_value[28] + value[29] * \
                 other_value[3] + value[28] * other_value[2] - value[3] * other_value[29]
    output[22] = value[5] * other_value[30] - value[30] * other_value[5] - value[31] * other_value[9] + value[1] * \
                 other_value[26] - value[9] * other_value[31] - value[26] * other_value[1]
    output[23] = -value[30] * other_value[4] - value[31] * other_value[8] + value[4] * other_value[30] - value[8] * \
                 other_value[31] + value[1] * other_value[27] - value[27] * other_value[1]
    output[24] = -value[28] * other_value[1] + value[7] * other_value[31] + value[31] * other_value[7] + value[30] * \
                 other_value[3] + value[1] * other_value[28] - value[3] * other_value[30]
    output[25] = -value[31] * other_value[6] - value[29] * other_value[1] + value[2] * other_value[30] + value[1] * \
                 other_value[29] - value[6] * other_value[31] - value[30] * other_value[2]
    output[26] = -value[5] * other_value[31] - value[31] * other_value[5]
    output[27] = -value[4] * other_value[31] - value[31] * other_value[4]
    output[28] = value[31] * other_value[3] + value[3] * other_value[31]
    output[29] = -value[31] * other_value[2] - value[2] * other_value[31]
    output[30] = value[1] * other_value[31] + value[31] * other_value[1]


@numba.cuda.jit(device=True)
def gp_device(value, other_value, output):
    output[0] = value[0] * other_value[0] + value[3] * other_value[3] + value[4] * other_value[4] - value[5] * \
                other_value[5] - value[6] * other_value[6] - value[7] * other_value[7] - value[8] * other_value[8] + \
                value[9] * other_value[9] - value[10] * other_value[10] - value[11] * other_value[11] + value[12] * \
                other_value[12] - value[13] * other_value[13] + value[14] * other_value[14] + value[15] * other_value[
                    15] + value[2] * other_value[2] - value[16] * other_value[16] + value[18] * other_value[18] - value[
                    19] * other_value[19] + value[20] * other_value[20] + value[21] * other_value[21] - value[22] * \
                other_value[22] + value[23] * other_value[23] + value[24] * other_value[24] + value[25] * other_value[
                    25] + value[26] * other_value[26] - value[27] * other_value[27] - value[28] * other_value[28] - \
                value[29] * other_value[29] - value[30] * other_value[30] - value[17] * other_value[17] + value[1] * \
                other_value[1] - value[31] * other_value[31]
    output[1] = -value[31] * other_value[30] + value[28] * other_value[24] - value[26] * other_value[22] + value[29] * \
                other_value[25] + value[20] * other_value[14] - value[30] * other_value[31] - value[2] * other_value[
                    6] - value[11] * other_value[17] - value[10] * other_value[16] - value[4] * other_value[8] - value[
                    19] * other_value[13] + value[12] * other_value[18] + value[14] * other_value[20] - value[17] * \
                other_value[11] + value[18] * other_value[12] - value[13] * other_value[19] - value[3] * other_value[
                    7] + value[27] * other_value[23] + value[21] * other_value[15] + value[5] * other_value[9] + value[
                    0] * other_value[1] - value[25] * other_value[29] + value[8] * other_value[4] + value[22] * \
                other_value[26] - value[9] * other_value[5] + value[15] * other_value[21] - value[24] * other_value[
                    28] - value[23] * other_value[27] + value[7] * other_value[3] + value[1] * other_value[0] + value[
                    6] * other_value[2] - value[16] * other_value[10]
    output[2] = value[23] * other_value[14] - value[18] * other_value[9] - value[13] * other_value[22] - value[3] * \
                other_value[10] + value[7] * other_value[16] + value[30] * other_value[25] - value[12] * other_value[
                    5] - value[25] * other_value[30] - value[4] * other_value[11] - value[27] * other_value[20] + value[
                    24] * other_value[15] + value[11] * other_value[4] - value[22] * other_value[13] + value[20] * \
                other_value[27] + value[26] * other_value[19] - value[9] * other_value[18] + value[5] * other_value[
                    12] + value[10] * other_value[3] - value[6] * other_value[1] + value[21] * other_value[28] + value[
                    8] * other_value[17] - value[28] * other_value[21] - value[19] * other_value[26] + value[31] * \
                other_value[29] + value[17] * other_value[8] + value[0] * other_value[2] + value[16] * other_value[7] + \
                value[15] * other_value[24] + value[14] * other_value[23] + value[2] * other_value[0] + value[29] * \
                other_value[31] + value[1] * other_value[6]
    output[3] = value[11] * other_value[22] + value[19] * other_value[8] - value[4] * other_value[13] - value[20] * \
                other_value[9] - value[26] * other_value[17] - value[30] * other_value[24] + value[2] * other_value[
                    10] + value[13] * other_value[4] + value[21] * other_value[29] - value[10] * other_value[2] + value[
                    25] * other_value[15] - value[16] * other_value[6] - value[6] * other_value[16] - value[9] * \
                other_value[20] + value[15] * other_value[25] + value[22] * other_value[11] + value[24] * other_value[
                    30] + value[8] * other_value[19] - value[7] * other_value[1] + value[1] * other_value[7] - value[
                    23] * other_value[12] - value[29] * other_value[21] + value[27] * other_value[18] + value[5] * \
                other_value[14] - value[28] * other_value[31] - value[31] * other_value[28] - value[14] * other_value[
                    5] + value[3] * other_value[0] - value[18] * other_value[27] + value[17] * other_value[26] - value[
                    12] * other_value[23] + value[0] * other_value[3]
    output[4] = -value[20] * other_value[29] + value[26] * other_value[16] - value[17] * other_value[6] + value[30] * \
                other_value[23] - value[13] * other_value[3] + value[5] * other_value[15] - value[10] * other_value[
                    22] + value[29] * other_value[20] - value[8] * other_value[1] - value[18] * other_value[28] - value[
                    21] * other_value[9] - value[25] * other_value[14] - value[22] * other_value[10] - value[16] * \
                other_value[26] + value[3] * other_value[13] - value[6] * other_value[17] - value[15] * other_value[5] - \
                value[24] * other_value[12] - value[14] * other_value[25] - value[9] * other_value[21] + value[0] * \
                other_value[4] + value[2] * other_value[11] - value[7] * other_value[19] - value[11] * other_value[2] - \
                value[12] * other_value[24] + value[27] * other_value[31] + value[1] * other_value[8] + value[28] * \
                other_value[18] + value[31] * other_value[27] - value[23] * other_value[30] - value[19] * other_value[
                    7] + value[4] * other_value[0]
    output[5] = -value[24] * other_value[11] - value[23] * other_value[10] + value[4] * other_value[15] + value[1] * \
                other_value[9] - value[8] * other_value[21] - value[17] * other_value[28] - value[25] * other_value[
                    13] + value[3] * other_value[14] - value[22] * other_value[30] - value[16] * other_value[27] - \
                value[21] * other_value[8] - value[12] * other_value[2] - value[15] * other_value[4] - value[14] * \
                other_value[3] - value[18] * other_value[6] - value[7] * other_value[20] + value[26] * other_value[31] - \
                value[11] * other_value[24] + value[27] * other_value[16] + value[0] * other_value[5] + value[31] * \
                other_value[26] - value[10] * other_value[23] + value[28] * other_value[17] + value[29] * other_value[
                    19] - value[19] * other_value[29] + value[5] * other_value[0] - value[20] * other_value[7] - value[
                    13] * other_value[25] - value[6] * other_value[18] + value[2] * other_value[12] - value[9] * \
                other_value[1] + value[30] * other_value[22]
    output[6] = value[17] * other_value[4] + value[28] * other_value[15] - value[8] * other_value[11] - value[24] * \
                other_value[21] - value[13] * other_value[26] + value[15] * other_value[28] + value[30] * other_value[
                    29] - value[7] * other_value[10] + value[1] * other_value[2] + value[22] * other_value[19] + value[
                    27] * other_value[14] + value[3] * other_value[16] - value[12] * other_value[9] + value[9] * \
                other_value[12] - value[5] * other_value[18] + value[10] * other_value[7] - value[29] * other_value[
                    30] + value[14] * other_value[27] - value[26] * other_value[13] - value[18] * other_value[5] + \
                value[21] * other_value[24] - value[19] * other_value[22] - value[23] * other_value[20] + value[25] * \
                other_value[31] + value[16] * other_value[3] + value[0] * other_value[6] + value[31] * other_value[25] + \
                value[6] * other_value[0] + value[4] * other_value[17] - value[2] * other_value[1] + value[20] * \
                other_value[23] + value[11] * other_value[8]
    output[7] = value[17] * other_value[22] - value[30] * other_value[28] + value[11] * other_value[26] + value[19] * \
                other_value[4] - value[20] * other_value[5] - value[18] * other_value[23] + value[15] * other_value[
                    29] - value[10] * other_value[6] - value[14] * other_value[9] - value[16] * other_value[2] + value[
                    21] * other_value[25] - value[22] * other_value[17] + value[9] * other_value[14] + value[13] * \
                other_value[8] - value[12] * other_value[27] + value[23] * other_value[18] - value[8] * other_value[
                    13] + value[26] * other_value[11] + value[6] * other_value[10] - value[25] * other_value[21] - \
                value[5] * other_value[20] - value[24] * other_value[31] - value[31] * other_value[24] + value[7] * \
                other_value[0] + value[4] * other_value[19] - value[27] * other_value[12] + value[0] * other_value[7] + \
                value[1] * other_value[3] - value[2] * other_value[16] + value[28] * other_value[30] + value[29] * \
                other_value[15] - value[3] * other_value[1]
    output[8] = value[30] * other_value[27] + value[6] * other_value[11] + value[9] * other_value[15] - value[21] * \
                other_value[5] + value[25] * other_value[20] - value[10] * other_value[26] - value[20] * other_value[
                    25] - value[5] * other_value[21] - value[29] * other_value[14] - value[3] * other_value[19] - value[
                    16] * other_value[22] - value[19] * other_value[3] - value[4] * other_value[1] - value[14] * \
                other_value[29] + value[1] * other_value[4] - value[17] * other_value[2] - value[13] * other_value[7] - \
                value[28] * other_value[12] - value[27] * other_value[30] - value[12] * other_value[28] - value[18] * \
                other_value[24] - value[26] * other_value[10] - value[11] * other_value[6] - value[2] * other_value[
                    17] + value[0] * other_value[8] - value[15] * other_value[9] + value[24] * other_value[18] + value[
                    8] * other_value[0] + value[7] * other_value[13] + value[31] * other_value[23] + value[23] * \
                other_value[31] + value[22] * other_value[16]
    output[9] = -value[5] * other_value[1] - value[21] * other_value[4] - value[10] * other_value[27] - value[15] * \
                other_value[8] + value[25] * other_value[19] - value[20] * other_value[3] - value[12] * other_value[6] + \
                value[7] * other_value[14] - value[4] * other_value[21] - value[27] * other_value[10] - value[2] * \
                other_value[18] - value[16] * other_value[23] + value[23] * other_value[16] - value[14] * other_value[
                    7] - value[17] * other_value[24] + value[30] * other_value[26] + value[8] * other_value[15] + value[
                    6] * other_value[12] - value[26] * other_value[30] + value[31] * other_value[22] + value[0] * \
                other_value[9] + value[1] * other_value[5] - value[13] * other_value[29] - value[3] * other_value[20] + \
                value[9] * other_value[0] - value[19] * other_value[25] - value[11] * other_value[28] + value[24] * \
                other_value[17] - value[18] * other_value[2] - value[28] * other_value[11] + value[22] * other_value[
                    31] - value[29] * other_value[13]
    output[10] = -value[11] * other_value[13] + value[16] * other_value[1] - value[26] * other_value[8] + value[29] * \
                 other_value[28] - value[20] * other_value[18] - value[14] * other_value[12] + value[19] * other_value[
                     17] + value[4] * other_value[22] + value[30] * other_value[15] - value[8] * other_value[26] - \
                 value[17] * other_value[19] - value[23] * other_value[5] + value[24] * other_value[25] + value[10] * \
                 other_value[0] + value[13] * other_value[11] + value[7] * other_value[6] + value[1] * other_value[16] + \
                 value[18] * other_value[20] - value[3] * other_value[2] - value[6] * other_value[7] + value[12] * \
                 other_value[14] - value[28] * other_value[29] + value[9] * other_value[27] + value[21] * other_value[
                     31] - value[25] * other_value[24] + value[22] * other_value[4] + value[0] * other_value[10] + \
                 value[31] * other_value[21] + value[2] * other_value[3] + value[27] * other_value[9] - value[5] * \
                 other_value[23] + value[15] * other_value[30]
    output[11] = value[28] * other_value[9] + value[7] * other_value[26] - value[19] * other_value[16] + value[18] * \
                 other_value[21] - value[30] * other_value[14] - value[13] * other_value[10] + value[12] * other_value[
                     15] - value[4] * other_value[2] + value[16] * other_value[19] - value[14] * other_value[30] + \
                 value[27] * other_value[29] - value[3] * other_value[22] - value[24] * other_value[5] + value[0] * \
                 other_value[11] + value[25] * other_value[23] + value[2] * other_value[4] - value[6] * other_value[8] - \
                 value[22] * other_value[3] + value[9] * other_value[28] + value[1] * other_value[17] - value[23] * \
                 other_value[25] - value[21] * other_value[18] - value[5] * other_value[24] + value[10] * other_value[
                     13] - value[20] * other_value[31] - value[15] * other_value[12] + value[8] * other_value[6] + \
                 value[26] * other_value[7] + value[11] * other_value[0] + value[17] * other_value[1] - value[29] * \
                 other_value[27] - value[31] * other_value[20]
    output[12] = value[18] * other_value[1] + value[1] * other_value[18] - value[23] * other_value[3] + value[9] * \
                 other_value[6] - value[19] * other_value[31] - value[3] * other_value[23] + value[26] * other_value[
                     29] - value[22] * other_value[25] + value[17] * other_value[21] - value[6] * other_value[9] - \
                 value[30] * other_value[13] - value[24] * other_value[4] + value[7] * other_value[27] - value[15] * \
                 other_value[11] - value[13] * other_value[30] + value[11] * other_value[15] + value[16] * other_value[
                     20] - value[20] * other_value[16] - value[29] * other_value[26] + value[28] * other_value[8] + \
                 value[0] * other_value[12] - value[31] * other_value[19] + value[10] * other_value[14] - value[4] * \
                 other_value[24] + value[8] * other_value[28] - value[5] * other_value[2] + value[2] * other_value[5] + \
                 value[25] * other_value[22] - value[21] * other_value[17] + value[12] * other_value[0] - value[14] * \
                 other_value[10] + value[27] * other_value[7]
    output[13] = value[30] * other_value[12] + value[28] * other_value[27] + value[2] * other_value[22] + value[17] * \
                 other_value[16] + value[13] * other_value[0] - value[15] * other_value[14] + value[31] * other_value[
                     18] - value[24] * other_value[23] - value[6] * other_value[26] + value[22] * other_value[2] + \
                 value[9] * other_value[29] - value[21] * other_value[20] - value[25] * other_value[5] + value[1] * \
                 other_value[19] - value[5] * other_value[25] + value[8] * other_value[7] - value[10] * other_value[
                     11] - value[7] * other_value[8] + value[20] * other_value[21] + value[29] * other_value[9] - value[
                     4] * other_value[3] + value[0] * other_value[13] + value[3] * other_value[4] + value[14] * \
                 other_value[15] + value[18] * other_value[31] - value[26] * other_value[6] - value[16] * other_value[
                     17] + value[12] * other_value[30] - value[27] * other_value[28] + value[19] * other_value[1] + \
                 value[11] * other_value[10] + value[23] * other_value[24]
    output[14] = value[1] * other_value[20] - value[16] * other_value[18] + value[14] * other_value[0] + value[29] * \
                 other_value[8] + value[30] * other_value[11] - value[15] * other_value[13] + value[0] * other_value[
                     14] - value[7] * other_value[9] + value[8] * other_value[29] + value[22] * other_value[24] - value[
                     24] * other_value[22] - value[6] * other_value[27] + value[9] * other_value[7] - value[21] * \
                 other_value[19] - value[25] * other_value[4] - value[5] * other_value[3] - value[10] * other_value[
                     12] + value[31] * other_value[17] - value[26] * other_value[28] - value[4] * other_value[25] + \
                 value[11] * other_value[30] + value[20] * other_value[1] + value[19] * other_value[21] + value[2] * \
                 other_value[23] + value[28] * other_value[26] + value[17] * other_value[31] + value[13] * other_value[
                     15] + value[23] * other_value[2] + value[3] * other_value[5] - value[27] * other_value[6] + value[
                     12] * other_value[10] + value[18] * other_value[16]
    output[15] = -value[29] * other_value[7] + value[23] * other_value[22] - value[28] * other_value[6] - value[22] * \
                 other_value[23] + value[26] * other_value[27] + value[24] * other_value[2] - value[17] * other_value[
                     18] + value[21] * other_value[1] + value[20] * other_value[19] - value[27] * other_value[26] + \
                 value[25] * other_value[3] - value[16] * other_value[31] + value[0] * other_value[15] + value[18] * \
                 other_value[17] - value[19] * other_value[20] + value[15] * other_value[0] - value[30] * other_value[
                     10] - value[13] * other_value[14] + value[12] * other_value[11] - value[11] * other_value[12] - \
                 value[10] * other_value[30] + value[14] * other_value[13] + value[9] * other_value[8] - value[8] * \
                 other_value[9] - value[6] * other_value[28] - value[5] * other_value[4] - value[31] * other_value[16] - \
                 value[7] * other_value[29] + value[1] * other_value[21] + value[3] * other_value[25] + value[2] * \
                 other_value[24] + value[4] * other_value[5]
    output[16] = -value[4] * other_value[26] - value[29] * other_value[24] - value[17] * other_value[13] + value[12] * \
                 other_value[20] + value[18] * other_value[14] - value[11] * other_value[19] + value[1] * other_value[
                     10] + value[19] * other_value[11] + value[10] * other_value[1] + value[28] * other_value[25] - \
                 value[20] * other_value[12] - value[9] * other_value[23] + value[21] * other_value[30] + value[8] * \
                 other_value[22] - value[27] * other_value[5] - value[22] * other_value[8] + value[0] * other_value[
                     16] - value[7] * other_value[2] + value[13] * other_value[17] + value[23] * other_value[9] + value[
                     6] * other_value[3] - value[24] * other_value[29] + value[5] * other_value[27] + value[3] * \
                 other_value[6] + value[26] * other_value[4] + value[25] * other_value[28] - value[2] * other_value[7] + \
                 value[31] * other_value[15] + value[15] * other_value[31] + value[16] * other_value[0] - value[14] * \
                 other_value[18] + value[30] * other_value[21]
    output[17] = value[16] * other_value[13] + value[18] * other_value[15] + value[1] * other_value[11] + value[11] * \
                 other_value[1] - value[7] * other_value[22] - value[31] * other_value[14] - value[2] * other_value[8] - \
                 value[26] * other_value[3] + value[5] * other_value[28] + value[23] * other_value[29] - value[25] * \
                 other_value[27] - value[28] * other_value[5] + value[10] * other_value[19] - value[8] * other_value[
                     2] + value[12] * other_value[21] + value[4] * other_value[6] - value[21] * other_value[12] - value[
                     19] * other_value[10] + value[3] * other_value[26] + value[0] * other_value[17] - value[30] * \
                 other_value[20] + value[6] * other_value[4] + value[24] * other_value[9] - value[20] * other_value[
                     30] - value[27] * other_value[25] + value[17] * other_value[0] + value[22] * other_value[7] + \
                 value[29] * other_value[23] - value[14] * other_value[31] - value[15] * other_value[18] - value[13] * \
                 other_value[16] - value[9] * other_value[24]
    output[18] = value[18] * other_value[0] + value[10] * other_value[20] - value[28] * other_value[4] - value[19] * \
                 other_value[30] + value[4] * other_value[28] - value[31] * other_value[13] + value[11] * other_value[
                     21] - value[20] * other_value[10] - value[21] * other_value[11] - value[27] * other_value[3] + \
                 value[3] * other_value[27] - value[2] * other_value[9] - value[26] * other_value[25] - value[9] * \
                 other_value[2] + value[16] * other_value[14] - value[14] * other_value[16] + value[6] * other_value[
                     5] + value[22] * other_value[29] + value[24] * other_value[8] + value[29] * other_value[22] + \
                 value[23] * other_value[7] - value[13] * other_value[31] - value[15] * other_value[17] + value[17] * \
                 other_value[15] + value[5] * other_value[6] + value[1] * other_value[12] - value[7] * other_value[23] + \
                 value[0] * other_value[18] - value[30] * other_value[19] - value[25] * other_value[26] - value[8] * \
                 other_value[24] + value[12] * other_value[1]
    output[19] = value[25] * other_value[9] + value[0] * other_value[19] - value[8] * other_value[3] - value[22] * \
                 other_value[6] + value[13] * other_value[1] - value[9] * other_value[25] - value[3] * other_value[8] + \
                 value[31] * other_value[12] + value[12] * other_value[31] - value[23] * other_value[28] + value[6] * \
                 other_value[22] + value[20] * other_value[15] + value[27] * other_value[24] + value[18] * other_value[
                     30] - value[29] * other_value[5] + value[24] * other_value[27] - value[10] * other_value[17] + \
                 value[14] * other_value[21] + value[17] * other_value[10] - value[16] * other_value[11] + value[7] * \
                 other_value[4] + value[5] * other_value[29] - value[28] * other_value[23] + value[30] * other_value[
                     18] - value[15] * other_value[20] + value[4] * other_value[7] + value[1] * other_value[13] + value[
                     26] * other_value[2] - value[2] * other_value[26] + value[19] * other_value[0] + value[11] * \
                 other_value[16] - value[21] * other_value[14]
    output[20] = value[5] * other_value[7] + value[25] * other_value[8] + value[7] * other_value[5] + value[6] * \
                 other_value[23] - value[22] * other_value[28] + value[24] * other_value[26] + value[26] * other_value[
                     24] - value[21] * other_value[13] - value[3] * other_value[9] + value[31] * other_value[11] - \
                 value[9] * other_value[3] + value[14] * other_value[1] + value[4] * other_value[29] - value[23] * \
                 other_value[6] - value[8] * other_value[25] + value[20] * other_value[0] + value[30] * other_value[
                     17] + value[12] * other_value[16] + value[11] * other_value[31] + value[1] * other_value[14] - \
                 value[28] * other_value[22] + value[0] * other_value[20] + value[18] * other_value[10] + value[17] * \
                 other_value[30] + value[13] * other_value[21] - value[10] * other_value[18] - value[2] * other_value[
                     27] - value[29] * other_value[4] + value[27] * other_value[2] + value[19] * other_value[15] - \
                 value[16] * other_value[12] - value[15] * other_value[19]
    output[21] = value[15] * other_value[1] - value[31] * other_value[10] + value[28] * other_value[2] - value[4] * \
                 other_value[9] + value[8] * other_value[5] + value[18] * other_value[11] - value[10] * other_value[
                     31] + value[22] * other_value[27] + value[12] * other_value[17] - value[25] * other_value[7] + \
                 value[7] * other_value[25] + value[1] * other_value[15] - value[17] * other_value[12] - value[13] * \
                 other_value[20] - value[23] * other_value[26] + value[5] * other_value[8] - value[24] * other_value[
                     6] + value[29] * other_value[3] + value[6] * other_value[24] + value[14] * other_value[19] - value[
                     26] * other_value[23] - value[16] * other_value[30] - value[3] * other_value[29] - value[11] * \
                 other_value[18] + value[27] * other_value[22] - value[19] * other_value[14] + value[20] * other_value[
                     13] + value[21] * other_value[0] - value[9] * other_value[4] - value[2] * other_value[28] - value[
                     30] * other_value[16] + value[0] * other_value[21]
    output[22] = -value[21] * other_value[27] + value[25] * other_value[12] - value[31] * other_value[9] + value[28] * \
                 other_value[20] + value[7] * other_value[17] + value[1] * other_value[26] + value[13] * other_value[
                     2] - value[27] * other_value[21] + value[22] * other_value[0] + value[23] * other_value[15] + \
                 value[20] * other_value[28] + value[5] * other_value[30] - value[24] * other_value[14] - value[29] * \
                 other_value[18] - value[6] * other_value[19] + value[10] * other_value[4] - value[17] * other_value[
                     7] - value[3] * other_value[11] - value[12] * other_value[25] - value[8] * other_value[16] - value[
                     15] * other_value[23] + value[19] * other_value[6] - value[30] * other_value[5] + value[2] * \
                 other_value[13] + value[4] * other_value[10] - value[11] * other_value[3] - value[18] * other_value[
                     29] - value[9] * other_value[31] + value[0] * other_value[22] - value[26] * other_value[1] + value[
                     16] * other_value[8] + value[14] * other_value[24]
    output[23] = -value[27] * other_value[1] + value[2] * other_value[14] + value[10] * other_value[5] - value[15] * \
                 other_value[22] + value[14] * other_value[2] + value[20] * other_value[6] - value[26] * other_value[
                     21] + value[16] * other_value[9] + value[5] * other_value[10] - value[11] * other_value[25] + \
                 value[13] * other_value[24] - value[3] * other_value[12] - value[24] * other_value[13] + value[19] * \
                 other_value[28] + value[23] * other_value[0] - value[17] * other_value[29] + value[28] * other_value[
                     19] + value[0] * other_value[23] - value[9] * other_value[16] - value[30] * other_value[4] + value[
                     25] * other_value[11] - value[12] * other_value[3] - value[21] * other_value[26] + value[4] * \
                 other_value[30] + value[22] * other_value[15] - value[29] * other_value[17] + value[7] * other_value[
                     18] - value[18] * other_value[7] - value[31] * other_value[8] + value[1] * other_value[27] - value[
                     8] * other_value[31] - value[6] * other_value[20]
    output[24] = value[29] * other_value[16] - value[3] * other_value[30] + value[31] * other_value[7] + value[5] * \
                 other_value[11] + value[15] * other_value[2] - value[27] * other_value[19] + value[1] * other_value[
                     28] - value[28] * other_value[1] + value[26] * other_value[20] - value[25] * other_value[10] - \
                 value[4] * other_value[12] + value[24] * other_value[0] + value[2] * other_value[15] + value[14] * \
                 other_value[22] - value[18] * other_value[8] + value[8] * other_value[18] + value[23] * other_value[
                     13] - value[9] * other_value[17] + value[21] * other_value[6] + value[17] * other_value[9] + value[
                     16] * other_value[29] - value[12] * other_value[4] - value[13] * other_value[23] - value[19] * \
                 other_value[27] + value[7] * other_value[31] + value[0] * other_value[24] - value[22] * other_value[
                     14] + value[30] * other_value[3] + value[11] * other_value[5] + value[20] * other_value[26] + \
                 value[10] * other_value[25] - value[6] * other_value[21]
    output[25] = -value[18] * other_value[26] - value[26] * other_value[18] - value[9] * other_value[19] + value[22] * \
                 other_value[12] + value[21] * other_value[7] + value[25] * other_value[0] - value[4] * other_value[
                     14] + value[11] * other_value[23] + value[19] * other_value[9] - value[12] * other_value[22] + \
                 value[1] * other_value[29] - value[31] * other_value[6] - value[28] * other_value[16] + value[15] * \
                 other_value[3] - value[14] * other_value[4] + value[0] * other_value[25] - value[20] * other_value[8] - \
                 value[6] * other_value[31] - value[29] * other_value[1] - value[10] * other_value[24] + value[24] * \
                 other_value[10] + value[3] * other_value[15] + value[13] * other_value[5] - value[7] * other_value[
                     21] - value[16] * other_value[28] - value[30] * other_value[2] + value[8] * other_value[20] + \
                 value[27] * other_value[17] + value[5] * other_value[13] - value[23] * other_value[11] + value[2] * \
                 other_value[30] + value[17] * other_value[27]
    output[26] = value[19] * other_value[2] - value[21] * other_value[23] + value[3] * other_value[17] - value[15] * \
                 other_value[27] + value[27] * other_value[15] + value[10] * other_value[8] - value[31] * other_value[
                     5] - value[11] * other_value[7] - value[2] * other_value[19] + value[26] * other_value[0] + value[
                     9] * other_value[30] + value[14] * other_value[28] + value[29] * other_value[12] + value[6] * \
                 other_value[13] + value[13] * other_value[6] + value[24] * other_value[20] + value[16] * other_value[
                     4] - value[5] * other_value[31] - value[17] * other_value[3] - value[23] * other_value[21] + value[
                     8] * other_value[10] - value[7] * other_value[11] + value[1] * other_value[22] + value[20] * \
                 other_value[24] - value[25] * other_value[18] - value[28] * other_value[14] - value[4] * other_value[
                     16] - value[12] * other_value[29] - value[18] * other_value[25] - value[22] * other_value[1] - \
                 value[30] * other_value[9] + value[0] * other_value[26]
    output[27] = value[29] * other_value[11] - value[18] * other_value[3] + value[16] * other_value[5] + value[13] * \
                 other_value[28] - value[28] * other_value[13] + value[1] * other_value[23] + value[27] * other_value[
                     0] + value[19] * other_value[24] - value[31] * other_value[4] - value[2] * other_value[20] - value[
                     17] * other_value[25] - value[12] * other_value[7] + value[10] * other_value[9] - value[11] * \
                 other_value[29] + value[14] * other_value[6] - value[25] * other_value[17] + value[8] * other_value[
                     30] + value[20] * other_value[2] - value[22] * other_value[21] - value[15] * other_value[26] - \
                 value[21] * other_value[22] + value[26] * other_value[15] - value[7] * other_value[12] - value[23] * \
                 other_value[1] - value[5] * other_value[16] - value[30] * other_value[8] + value[0] * other_value[27] - \
                 value[4] * other_value[31] + value[24] * other_value[19] + value[6] * other_value[14] + value[9] * \
                 other_value[10] + value[3] * other_value[18]
    output[28] = value[25] * other_value[16] - value[29] * other_value[10] + value[14] * other_value[26] + value[22] * \
                 other_value[20] + value[28] * other_value[0] + value[31] * other_value[3] - value[23] * other_value[
                     19] + value[1] * other_value[24] - value[7] * other_value[30] - value[13] * other_value[27] - \
                 value[12] * other_value[8] + value[17] * other_value[5] + value[4] * other_value[18] + value[16] * \
                 other_value[25] - value[24] * other_value[1] - value[5] * other_value[17] + value[6] * other_value[
                     15] + value[20] * other_value[22] + value[21] * other_value[2] + value[30] * other_value[7] - \
                 value[18] * other_value[4] + value[3] * other_value[31] - value[2] * other_value[21] - value[19] * \
                 other_value[23] - value[26] * other_value[14] + value[0] * other_value[28] + value[10] * other_value[
                     29] + value[27] * other_value[13] + value[11] * other_value[9] + value[15] * other_value[6] - \
                 value[8] * other_value[12] + value[9] * other_value[11]
    output[29] = value[29] * other_value[0] - value[3] * other_value[21] - value[20] * other_value[4] - value[5] * \
                 other_value[19] + value[9] * other_value[13] + value[13] * other_value[9] - value[14] * other_value[
                     8] + value[19] * other_value[5] + value[23] * other_value[17] + value[6] * other_value[30] - value[
                     10] * other_value[28] - value[24] * other_value[16] - value[18] * other_value[22] - value[12] * \
                 other_value[26] + value[17] * other_value[23] + value[11] * other_value[27] + value[4] * other_value[
                     20] - value[25] * other_value[1] - value[8] * other_value[14] - value[22] * other_value[18] + \
                 value[7] * other_value[15] + value[21] * other_value[3] + value[28] * other_value[10] + value[0] * \
                 other_value[29] - value[30] * other_value[6] + value[1] * other_value[25] + value[26] * other_value[
                     12] + value[15] * other_value[7] - value[31] * other_value[2] - value[2] * other_value[31] - value[
                     27] * other_value[11] - value[16] * other_value[24]
    output[30] = value[0] * other_value[30] - value[28] * other_value[7] + value[1] * other_value[31] + value[2] * \
                 other_value[25] - value[3] * other_value[24] + value[4] * other_value[23] - value[5] * other_value[
                     22] - value[6] * other_value[29] + value[7] * other_value[28] - value[8] * other_value[27] + value[
                     9] * other_value[26] + value[10] * other_value[15] - value[11] * other_value[14] + value[12] * \
                 other_value[13] + value[13] * other_value[12] + value[29] * other_value[6] - value[14] * other_value[
                     11] + value[15] * other_value[10] + value[16] * other_value[21] - value[17] * other_value[20] + \
                 value[18] * other_value[19] + value[19] * other_value[18] - value[20] * other_value[17] + value[21] * \
                 other_value[16] + value[22] * other_value[5] - value[23] * other_value[4] + value[24] * other_value[
                     3] - value[25] * other_value[2] - value[26] * other_value[9] + value[27] * other_value[8] + value[
                     30] * other_value[0] + value[31] * other_value[1]
    output[31] = value[15] * other_value[16] - value[29] * other_value[2] + value[0] * other_value[31] + value[1] * \
                 other_value[30] - value[2] * other_value[29] + value[3] * other_value[28] - value[4] * other_value[
                     27] + value[5] * other_value[26] + value[6] * other_value[25] - value[7] * other_value[24] + value[
                     8] * other_value[23] - value[9] * other_value[22] + value[10] * other_value[21] - value[11] * \
                 other_value[20] + value[12] * other_value[19] + value[30] * other_value[1] + value[13] * other_value[
                     18] + value[16] * other_value[15] - value[17] * other_value[14] + value[18] * other_value[13] + \
                 value[19] * other_value[12] - value[20] * other_value[11] + value[21] * other_value[10] - value[22] * \
                 other_value[9] + value[23] * other_value[8] - value[24] * other_value[7] + value[25] * other_value[6] + \
                 value[26] * other_value[5] - value[27] * other_value[4] + value[28] * other_value[3] - value[14] * \
                 other_value[17] + value[31] * other_value[0]

