import numpy as np
from numba import njit, cuda


@cuda.jit(device=True)
def gmt_func_o0(value, other_value):
    return 0 + 1.0*value[0]*other_value[0] + 1.0*value[3]*other_value[3] + 1.0*value[4]*other_value[4] + -1.0*value[5]*other_value[5] + -1.0*value[6]*other_value[6] + -1.0*value[7]*other_value[7] + -1.0*value[8]*other_value[8] + 1.0*value[9]*other_value[9] + -1.0*value[10]*other_value[10] + -1.0*value[11]*other_value[11] + 1.0*value[12]*other_value[12] + -1.0*value[13]*other_value[13] + 1.0*value[14]*other_value[14] + 1.0*value[15]*other_value[15] + 1.0*value[2]*other_value[2] + -1.0*value[16]*other_value[16] + 1.0*value[18]*other_value[18] + -1.0*value[19]*other_value[19] + 1.0*value[20]*other_value[20] + 1.0*value[21]*other_value[21] + -1.0*value[22]*other_value[22] + 1.0*value[23]*other_value[23] + 1.0*value[24]*other_value[24] + 1.0*value[25]*other_value[25] + 1.0*value[26]*other_value[26] + -1.0*value[27]*other_value[27] + -1.0*value[28]*other_value[28] + -1.0*value[29]*other_value[29] + -1.0*value[30]*other_value[30] + -1.0*value[17]*other_value[17] + 1.0*value[1]*other_value[1] + -1.0*value[31]*other_value[31]

@cuda.jit(device=True)
def gmt_func_o1(value, other_value):
    return 0 + 1.0*value[1]*other_value[0] + -1.0*value[23]*other_value[27] + 1.0*value[20]*other_value[14] + 1.0*value[14]*other_value[20] + 1.0*value[18]*other_value[12] + -1.0*value[30]*other_value[31] + 1.0*value[0]*other_value[1] + 1.0*value[22]*other_value[26] + -1.0*value[10]*other_value[16] + 1.0*value[29]*other_value[25] + -1.0*value[2]*other_value[6] + -1.0*value[13]*other_value[19] + -1.0*value[3]*other_value[7] + -1.0*value[11]*other_value[17] + -1.0*value[19]*other_value[13] + 1.0*value[21]*other_value[15] + -1.0*value[4]*other_value[8] + 1.0*value[28]*other_value[24] + -1.0*value[9]*other_value[5] + 1.0*value[12]*other_value[18] + 1.0*value[5]*other_value[9] + -1.0*value[17]*other_value[11] + -1.0*value[26]*other_value[22] + -1.0*value[25]*other_value[29] + 1.0*value[7]*other_value[3] + -1.0*value[16]*other_value[10] + 1.0*value[6]*other_value[2] + -1.0*value[31]*other_value[30] + -1.0*value[24]*other_value[28] + 1.0*value[15]*other_value[21] + 1.0*value[8]*other_value[4] + 1.0*value[27]*other_value[23]

@cuda.jit(device=True)
def gmt_func_o2(value, other_value):
    return 0 + 1.0*value[11]*other_value[4] + 1.0*value[29]*other_value[31] + 1.0*value[5]*other_value[12] + -1.0*value[27]*other_value[20] + 1.0*value[20]*other_value[27] + -1.0*value[12]*other_value[5] + 1.0*value[0]*other_value[2] + 1.0*value[24]*other_value[15] + -1.0*value[6]*other_value[1] + 1.0*value[26]*other_value[19] + 1.0*value[10]*other_value[3] + -1.0*value[25]*other_value[30] + -1.0*value[9]*other_value[18] + 1.0*value[23]*other_value[14] + 1.0*value[7]*other_value[16] + -1.0*value[4]*other_value[11] + 1.0*value[8]*other_value[17] + -1.0*value[28]*other_value[21] + 1.0*value[21]*other_value[28] + -1.0*value[22]*other_value[13] + -1.0*value[3]*other_value[10] + 1.0*value[14]*other_value[23] + 1.0*value[30]*other_value[25] + -1.0*value[13]*other_value[22] + -1.0*value[18]*other_value[9] + 1.0*value[15]*other_value[24] + 1.0*value[1]*other_value[6] + 1.0*value[17]*other_value[8] + -1.0*value[19]*other_value[26] + 1.0*value[2]*other_value[0] + 1.0*value[16]*other_value[7] + 1.0*value[31]*other_value[29]

@cuda.jit(device=True)
def gmt_func_o3(value, other_value):
    return 0 + -1.0*value[9]*other_value[20] + -1.0*value[23]*other_value[12] + 1.0*value[0]*other_value[3] + 1.0*value[17]*other_value[26] + -1.0*value[4]*other_value[13] + 1.0*value[8]*other_value[19] + -1.0*value[28]*other_value[31] + -1.0*value[14]*other_value[5] + 1.0*value[24]*other_value[30] + 1.0*value[25]*other_value[15] + 1.0*value[15]*other_value[25] + -1.0*value[31]*other_value[28] + -1.0*value[7]*other_value[1] + -1.0*value[6]*other_value[16] + -1.0*value[26]*other_value[17] + 1.0*value[1]*other_value[7] + -1.0*value[16]*other_value[6] + 1.0*value[27]*other_value[18] + 1.0*value[5]*other_value[14] + -1.0*value[10]*other_value[2] + -1.0*value[18]*other_value[27] + -1.0*value[12]*other_value[23] + 1.0*value[22]*other_value[11] + -1.0*value[30]*other_value[24] + 1.0*value[11]*other_value[22] + 1.0*value[3]*other_value[0] + 1.0*value[2]*other_value[10] + 1.0*value[21]*other_value[29] + -1.0*value[29]*other_value[21] + 1.0*value[19]*other_value[8] + 1.0*value[13]*other_value[4] + -1.0*value[20]*other_value[9]

@cuda.jit(device=True)
def gmt_func_o4(value, other_value):
    return 0 + 1.0*value[0]*other_value[4] + -1.0*value[17]*other_value[6] + 1.0*value[2]*other_value[11] + -1.0*value[18]*other_value[28] + -1.0*value[21]*other_value[9] + -1.0*value[25]*other_value[14] + -1.0*value[7]*other_value[19] + 1.0*value[31]*other_value[27] + 1.0*value[27]*other_value[31] + 1.0*value[5]*other_value[15] + -1.0*value[13]*other_value[3] + -1.0*value[16]*other_value[26] + -1.0*value[6]*other_value[17] + 1.0*value[30]*other_value[23] + -1.0*value[20]*other_value[29] + 1.0*value[1]*other_value[8] + 1.0*value[26]*other_value[16] + -1.0*value[8]*other_value[1] + -1.0*value[15]*other_value[5] + 1.0*value[3]*other_value[13] + 1.0*value[28]*other_value[18] + -1.0*value[9]*other_value[21] + -1.0*value[12]*other_value[24] + -1.0*value[11]*other_value[2] + -1.0*value[23]*other_value[30] + 1.0*value[29]*other_value[20] + -1.0*value[14]*other_value[25] + 1.0*value[4]*other_value[0] + -1.0*value[24]*other_value[12] + -1.0*value[22]*other_value[10] + -1.0*value[19]*other_value[7] + -1.0*value[10]*other_value[22]

@cuda.jit(device=True)
def gmt_func_o5(value, other_value):
    return 0 + -1.0*value[19]*other_value[29] + -1.0*value[22]*other_value[30] + -1.0*value[14]*other_value[3] + -1.0*value[23]*other_value[10] + 1.0*value[0]*other_value[5] + -1.0*value[20]*other_value[7] + -1.0*value[12]*other_value[2] + -1.0*value[6]*other_value[18] + -1.0*value[11]*other_value[24] + -1.0*value[18]*other_value[6] + -1.0*value[16]*other_value[27] + -1.0*value[9]*other_value[1] + -1.0*value[21]*other_value[8] + -1.0*value[7]*other_value[20] + -1.0*value[8]*other_value[21] + -1.0*value[25]*other_value[13] + -1.0*value[17]*other_value[28] + -1.0*value[13]*other_value[25] + -1.0*value[15]*other_value[4] + -1.0*value[10]*other_value[23] + 1.0*value[26]*other_value[31] + -1.0*value[24]*other_value[11] + 1.0*value[27]*other_value[16] + 1.0*value[30]*other_value[22] + 1.0*value[3]*other_value[14] + 1.0*value[1]*other_value[9] + 1.0*value[31]*other_value[26] + 1.0*value[5]*other_value[0] + 1.0*value[4]*other_value[15] + 1.0*value[29]*other_value[19] + 1.0*value[28]*other_value[17] + 1.0*value[2]*other_value[12]

@cuda.jit(device=True)
def gmt_func_o6(value, other_value):
    return 0 + 1.0*value[11]*other_value[8] + 1.0*value[3]*other_value[16] + 1.0*value[10]*other_value[7] + -1.0*value[26]*other_value[13] + -1.0*value[29]*other_value[30] + 1.0*value[17]*other_value[4] + -1.0*value[19]*other_value[22] + 1.0*value[1]*other_value[2] + 1.0*value[25]*other_value[31] + 1.0*value[4]*other_value[17] + -1.0*value[12]*other_value[9] + 1.0*value[0]*other_value[6] + 1.0*value[22]*other_value[19] + 1.0*value[20]*other_value[23] + 1.0*value[21]*other_value[24] + -1.0*value[7]*other_value[10] + -1.0*value[23]*other_value[20] + 1.0*value[6]*other_value[0] + 1.0*value[28]*other_value[15] + 1.0*value[31]*other_value[25] + -1.0*value[5]*other_value[18] + -1.0*value[18]*other_value[5] + -1.0*value[2]*other_value[1] + 1.0*value[9]*other_value[12] + 1.0*value[27]*other_value[14] + 1.0*value[30]*other_value[29] + -1.0*value[24]*other_value[21] + -1.0*value[13]*other_value[26] + 1.0*value[14]*other_value[27] + -1.0*value[8]*other_value[11] + 1.0*value[16]*other_value[3] + 1.0*value[15]*other_value[28]

@cuda.jit(device=True)
def gmt_func_o7(value, other_value):
    return 0 + -1.0*value[25]*other_value[21] + 1.0*value[4]*other_value[19] + -1.0*value[20]*other_value[5] + 1.0*value[21]*other_value[25] + -1.0*value[8]*other_value[13] + 1.0*value[11]*other_value[26] + -1.0*value[24]*other_value[31] + -1.0*value[3]*other_value[1] + -1.0*value[16]*other_value[2] + 1.0*value[28]*other_value[30] + 1.0*value[17]*other_value[22] + -1.0*value[31]*other_value[24] + -1.0*value[18]*other_value[23] + -1.0*value[14]*other_value[9] + 1.0*value[6]*other_value[10] + 1.0*value[15]*other_value[29] + 1.0*value[19]*other_value[4] + -1.0*value[30]*other_value[28] + -1.0*value[12]*other_value[27] + -1.0*value[10]*other_value[6] + 1.0*value[26]*other_value[11] + 1.0*value[23]*other_value[18] + -1.0*value[5]*other_value[20] + 1.0*value[13]*other_value[8] + -1.0*value[2]*other_value[16] + 1.0*value[1]*other_value[3] + 1.0*value[0]*other_value[7] + 1.0*value[7]*other_value[0] + -1.0*value[22]*other_value[17] + 1.0*value[9]*other_value[14] + 1.0*value[29]*other_value[15] + -1.0*value[27]*other_value[12]

@cuda.jit(device=True)
def gmt_func_o8(value, other_value):
    return 0 + -1.0*value[10]*other_value[26] + 1.0*value[9]*other_value[15] + -1.0*value[17]*other_value[2] + -1.0*value[29]*other_value[14] + 1.0*value[24]*other_value[18] + 1.0*value[23]*other_value[31] + 1.0*value[22]*other_value[16] + -1.0*value[3]*other_value[19] + 1.0*value[0]*other_value[8] + -1.0*value[20]*other_value[25] + -1.0*value[4]*other_value[1] + 1.0*value[31]*other_value[23] + -1.0*value[14]*other_value[29] + -1.0*value[18]*other_value[24] + 1.0*value[6]*other_value[11] + -1.0*value[19]*other_value[3] + 1.0*value[30]*other_value[27] + -1.0*value[15]*other_value[9] + -1.0*value[26]*other_value[10] + -1.0*value[13]*other_value[7] + -1.0*value[2]*other_value[17] + -1.0*value[27]*other_value[30] + 1.0*value[1]*other_value[4] + -1.0*value[16]*other_value[22] + -1.0*value[28]*other_value[12] + -1.0*value[5]*other_value[21] + 1.0*value[7]*other_value[13] + 1.0*value[25]*other_value[20] + 1.0*value[8]*other_value[0] + -1.0*value[11]*other_value[6] + -1.0*value[12]*other_value[28] + -1.0*value[21]*other_value[5]

@cuda.jit(device=True)
def gmt_func_o9(value, other_value):
    return 0 + 1.0*value[31]*other_value[22] + -1.0*value[18]*other_value[2] + -1.0*value[14]*other_value[7] + -1.0*value[15]*other_value[8] + -1.0*value[13]*other_value[29] + 1.0*value[22]*other_value[31] + -1.0*value[3]*other_value[20] + 1.0*value[30]*other_value[26] + -1.0*value[21]*other_value[4] + -1.0*value[11]*other_value[28] + 1.0*value[0]*other_value[9] + -1.0*value[10]*other_value[27] + -1.0*value[12]*other_value[6] + -1.0*value[20]*other_value[3] + -1.0*value[19]*other_value[25] + -1.0*value[2]*other_value[18] + -1.0*value[29]*other_value[13] + -1.0*value[27]*other_value[10] + -1.0*value[17]*other_value[24] + 1.0*value[23]*other_value[16] + -1.0*value[4]*other_value[21] + 1.0*value[1]*other_value[5] + 1.0*value[9]*other_value[0] + -1.0*value[26]*other_value[30] + 1.0*value[8]*other_value[15] + -1.0*value[28]*other_value[11] + 1.0*value[25]*other_value[19] + 1.0*value[7]*other_value[14] + 1.0*value[6]*other_value[12] + 1.0*value[24]*other_value[17] + -1.0*value[16]*other_value[23] + -1.0*value[5]*other_value[1]

@cuda.jit(device=True)
def gmt_func_o10(value, other_value):
    return 0 + 1.0*value[22]*other_value[4] + -1.0*value[17]*other_value[19] + -1.0*value[14]*other_value[12] + 1.0*value[21]*other_value[31] + 1.0*value[16]*other_value[1] + 1.0*value[24]*other_value[25] + 1.0*value[13]*other_value[11] + 1.0*value[4]*other_value[22] + 1.0*value[0]*other_value[10] + 1.0*value[12]*other_value[14] + 1.0*value[18]*other_value[20] + 1.0*value[1]*other_value[16] + 1.0*value[7]*other_value[6] + 1.0*value[29]*other_value[28] + -1.0*value[25]*other_value[24] + -1.0*value[6]*other_value[7] + 1.0*value[2]*other_value[3] + 1.0*value[19]*other_value[17] + -1.0*value[26]*other_value[8] + 1.0*value[30]*other_value[15] + -1.0*value[20]*other_value[18] + -1.0*value[3]*other_value[2] + -1.0*value[11]*other_value[13] + 1.0*value[27]*other_value[9] + -1.0*value[28]*other_value[29] + 1.0*value[31]*other_value[21] + 1.0*value[10]*other_value[0] + -1.0*value[8]*other_value[26] + -1.0*value[5]*other_value[23] + 1.0*value[9]*other_value[27] + 1.0*value[15]*other_value[30] + -1.0*value[23]*other_value[5]

@cuda.jit(device=True)
def gmt_func_o11(value, other_value):
    return 0 + 1.0*value[16]*other_value[19] + 1.0*value[7]*other_value[26] + 1.0*value[27]*other_value[29] + -1.0*value[31]*other_value[20] + 1.0*value[26]*other_value[7] + 1.0*value[1]*other_value[17] + -1.0*value[15]*other_value[12] + -1.0*value[20]*other_value[31] + -1.0*value[6]*other_value[8] + 1.0*value[12]*other_value[15] + 1.0*value[9]*other_value[28] + 1.0*value[28]*other_value[9] + 1.0*value[2]*other_value[4] + -1.0*value[30]*other_value[14] + -1.0*value[21]*other_value[18] + 1.0*value[25]*other_value[23] + -1.0*value[23]*other_value[25] + -1.0*value[24]*other_value[5] + -1.0*value[22]*other_value[3] + -1.0*value[13]*other_value[10] + -1.0*value[3]*other_value[22] + -1.0*value[14]*other_value[30] + -1.0*value[5]*other_value[24] + 1.0*value[11]*other_value[0] + 1.0*value[8]*other_value[6] + -1.0*value[4]*other_value[2] + 1.0*value[17]*other_value[1] + 1.0*value[10]*other_value[13] + -1.0*value[19]*other_value[16] + 1.0*value[18]*other_value[21] + -1.0*value[29]*other_value[27] + 1.0*value[0]*other_value[11]

@cuda.jit(device=True)
def gmt_func_o12(value, other_value):
    return 0 + 1.0*value[0]*other_value[12] + 1.0*value[2]*other_value[5] + -1.0*value[23]*other_value[3] + -1.0*value[20]*other_value[16] + -1.0*value[19]*other_value[31] + 1.0*value[16]*other_value[20] + 1.0*value[18]*other_value[1] + 1.0*value[25]*other_value[22] + -1.0*value[5]*other_value[2] + 1.0*value[1]*other_value[18] + -1.0*value[6]*other_value[9] + -1.0*value[24]*other_value[4] + 1.0*value[8]*other_value[28] + -1.0*value[14]*other_value[10] + 1.0*value[10]*other_value[14] + -1.0*value[29]*other_value[26] + 1.0*value[7]*other_value[27] + -1.0*value[31]*other_value[19] + -1.0*value[15]*other_value[11] + 1.0*value[11]*other_value[15] + 1.0*value[9]*other_value[6] + -1.0*value[21]*other_value[17] + 1.0*value[27]*other_value[7] + -1.0*value[22]*other_value[25] + -1.0*value[3]*other_value[23] + -1.0*value[4]*other_value[24] + 1.0*value[28]*other_value[8] + -1.0*value[13]*other_value[30] + 1.0*value[17]*other_value[21] + -1.0*value[30]*other_value[13] + 1.0*value[26]*other_value[29] + 1.0*value[12]*other_value[0]

@cuda.jit(device=True)
def gmt_func_o13(value, other_value):
    return 0 + 1.0*value[0]*other_value[13] + 1.0*value[19]*other_value[1] + 1.0*value[1]*other_value[19] + 1.0*value[31]*other_value[18] + -1.0*value[5]*other_value[25] + -1.0*value[27]*other_value[28] + 1.0*value[18]*other_value[31] + 1.0*value[14]*other_value[15] + -1.0*value[10]*other_value[11] + 1.0*value[20]*other_value[21] + 1.0*value[8]*other_value[7] + 1.0*value[11]*other_value[10] + -1.0*value[21]*other_value[20] + -1.0*value[26]*other_value[6] + -1.0*value[16]*other_value[17] + 1.0*value[3]*other_value[4] + -1.0*value[25]*other_value[5] + 1.0*value[28]*other_value[27] + 1.0*value[30]*other_value[12] + 1.0*value[22]*other_value[2] + 1.0*value[17]*other_value[16] + -1.0*value[15]*other_value[14] + -1.0*value[24]*other_value[23] + -1.0*value[4]*other_value[3] + -1.0*value[6]*other_value[26] + 1.0*value[2]*other_value[22] + 1.0*value[9]*other_value[29] + 1.0*value[13]*other_value[0] + -1.0*value[7]*other_value[8] + 1.0*value[12]*other_value[30] + 1.0*value[23]*other_value[24] + 1.0*value[29]*other_value[9]

@cuda.jit(device=True)
def gmt_func_o14(value, other_value):
    return 0 + -1.0*value[24]*other_value[22] + 1.0*value[29]*other_value[8] + 1.0*value[28]*other_value[26] + -1.0*value[10]*other_value[12] + 1.0*value[17]*other_value[31] + 1.0*value[1]*other_value[20] + 1.0*value[31]*other_value[17] + 1.0*value[8]*other_value[29] + 1.0*value[23]*other_value[2] + 1.0*value[9]*other_value[7] + 1.0*value[22]*other_value[24] + 1.0*value[18]*other_value[16] + 1.0*value[19]*other_value[21] + -1.0*value[21]*other_value[19] + -1.0*value[16]*other_value[18] + -1.0*value[6]*other_value[27] + 1.0*value[13]*other_value[15] + -1.0*value[7]*other_value[9] + 1.0*value[20]*other_value[1] + -1.0*value[5]*other_value[3] + 1.0*value[2]*other_value[23] + -1.0*value[27]*other_value[6] + -1.0*value[15]*other_value[13] + 1.0*value[30]*other_value[11] + 1.0*value[11]*other_value[30] + -1.0*value[4]*other_value[25] + 1.0*value[14]*other_value[0] + -1.0*value[25]*other_value[4] + -1.0*value[26]*other_value[28] + 1.0*value[3]*other_value[5] + 1.0*value[0]*other_value[14] + 1.0*value[12]*other_value[10]

@cuda.jit(device=True)
def gmt_func_o15(value, other_value):
    return 0 + -1.0*value[17]*other_value[18] + -1.0*value[16]*other_value[31] + 1.0*value[0]*other_value[15] + 1.0*value[15]*other_value[0] + 1.0*value[3]*other_value[25] + -1.0*value[29]*other_value[7] + 1.0*value[23]*other_value[22] + 1.0*value[9]*other_value[8] + 1.0*value[4]*other_value[5] + 1.0*value[24]*other_value[2] + 1.0*value[1]*other_value[21] + -1.0*value[8]*other_value[9] + 1.0*value[25]*other_value[3] + -1.0*value[28]*other_value[6] + -1.0*value[6]*other_value[28] + -1.0*value[5]*other_value[4] + -1.0*value[27]*other_value[26] + 1.0*value[26]*other_value[27] + -1.0*value[7]*other_value[29] + -1.0*value[10]*other_value[30] + -1.0*value[22]*other_value[23] + -1.0*value[19]*other_value[20] + 1.0*value[21]*other_value[1] + -1.0*value[30]*other_value[10] + -1.0*value[31]*other_value[16] + 1.0*value[12]*other_value[11] + 1.0*value[20]*other_value[19] + 1.0*value[2]*other_value[24] + 1.0*value[18]*other_value[17] + 1.0*value[14]*other_value[13] + -1.0*value[11]*other_value[12] + -1.0*value[13]*other_value[14]

@cuda.jit(device=True)
def gmt_func_o16(value, other_value):
    return 0 + -1.0*value[27]*other_value[5] + 1.0*value[18]*other_value[14] + 1.0*value[26]*other_value[4] + 1.0*value[31]*other_value[15] + 1.0*value[25]*other_value[28] + 1.0*value[21]*other_value[30] + -1.0*value[24]*other_value[29] + 1.0*value[28]*other_value[25] + 1.0*value[30]*other_value[21] + -1.0*value[20]*other_value[12] + 1.0*value[23]*other_value[9] + -1.0*value[17]*other_value[13] + -1.0*value[22]*other_value[8] + 1.0*value[19]*other_value[11] + -1.0*value[29]*other_value[24] + 1.0*value[15]*other_value[31] + 1.0*value[0]*other_value[16] + 1.0*value[8]*other_value[22] + 1.0*value[12]*other_value[20] + -1.0*value[11]*other_value[19] + 1.0*value[3]*other_value[6] + -1.0*value[7]*other_value[2] + 1.0*value[13]*other_value[17] + -1.0*value[9]*other_value[23] + 1.0*value[16]*other_value[0] + 1.0*value[6]*other_value[3] + -1.0*value[4]*other_value[26] + 1.0*value[5]*other_value[27] + 1.0*value[10]*other_value[1] + -1.0*value[2]*other_value[7] + 1.0*value[1]*other_value[10] + -1.0*value[14]*other_value[18]

@cuda.jit(device=True)
def gmt_func_o17(value, other_value):
    return 0 + 1.0*value[0]*other_value[17] + 1.0*value[3]*other_value[26] + -1.0*value[7]*other_value[22] + -1.0*value[19]*other_value[10] + 1.0*value[12]*other_value[21] + 1.0*value[4]*other_value[6] + 1.0*value[1]*other_value[11] + 1.0*value[5]*other_value[28] + -1.0*value[25]*other_value[27] + 1.0*value[18]*other_value[15] + -1.0*value[26]*other_value[3] + 1.0*value[6]*other_value[4] + -1.0*value[28]*other_value[5] + -1.0*value[27]*other_value[25] + -1.0*value[14]*other_value[31] + -1.0*value[13]*other_value[16] + 1.0*value[29]*other_value[23] + -1.0*value[20]*other_value[30] + -1.0*value[9]*other_value[24] + 1.0*value[23]*other_value[29] + 1.0*value[22]*other_value[7] + -1.0*value[21]*other_value[12] + -1.0*value[30]*other_value[20] + 1.0*value[17]*other_value[0] + 1.0*value[24]*other_value[9] + -1.0*value[2]*other_value[8] + -1.0*value[15]*other_value[18] + 1.0*value[10]*other_value[19] + 1.0*value[16]*other_value[13] + -1.0*value[8]*other_value[2] + -1.0*value[31]*other_value[14] + 1.0*value[11]*other_value[1]

@cuda.jit(device=True)
def gmt_func_o18(value, other_value):
    return 0 + 1.0*value[22]*other_value[29] + 1.0*value[10]*other_value[20] + -1.0*value[13]*other_value[31] + -1.0*value[27]*other_value[3] + 1.0*value[23]*other_value[7] + -1.0*value[26]*other_value[25] + 1.0*value[0]*other_value[18] + 1.0*value[6]*other_value[5] + -1.0*value[19]*other_value[30] + -1.0*value[8]*other_value[24] + 1.0*value[24]*other_value[8] + -1.0*value[21]*other_value[11] + -1.0*value[25]*other_value[26] + -1.0*value[20]*other_value[10] + 1.0*value[11]*other_value[21] + -1.0*value[7]*other_value[23] + -1.0*value[31]*other_value[13] + 1.0*value[12]*other_value[1] + -1.0*value[9]*other_value[2] + 1.0*value[5]*other_value[6] + 1.0*value[16]*other_value[14] + -1.0*value[28]*other_value[4] + 1.0*value[29]*other_value[22] + 1.0*value[4]*other_value[28] + 1.0*value[3]*other_value[27] + 1.0*value[17]*other_value[15] + 1.0*value[1]*other_value[12] + -1.0*value[15]*other_value[17] + 1.0*value[18]*other_value[0] + -1.0*value[30]*other_value[19] + -1.0*value[14]*other_value[16] + -1.0*value[2]*other_value[9]

@cuda.jit(device=True)
def gmt_func_o19(value, other_value):
    return 0 + 1.0*value[7]*other_value[4] + -1.0*value[29]*other_value[5] + 1.0*value[20]*other_value[15] + 1.0*value[25]*other_value[9] + 1.0*value[31]*other_value[12] + 1.0*value[12]*other_value[31] + 1.0*value[26]*other_value[2] + 1.0*value[4]*other_value[7] + 1.0*value[24]*other_value[27] + 1.0*value[13]*other_value[1] + 1.0*value[17]*other_value[10] + -1.0*value[15]*other_value[20] + -1.0*value[21]*other_value[14] + -1.0*value[3]*other_value[8] + 1.0*value[11]*other_value[16] + -1.0*value[23]*other_value[28] + -1.0*value[2]*other_value[26] + 1.0*value[30]*other_value[18] + -1.0*value[9]*other_value[25] + -1.0*value[22]*other_value[6] + -1.0*value[16]*other_value[11] + -1.0*value[8]*other_value[3] + 1.0*value[6]*other_value[22] + 1.0*value[0]*other_value[19] + 1.0*value[19]*other_value[0] + 1.0*value[14]*other_value[21] + 1.0*value[27]*other_value[24] + -1.0*value[10]*other_value[17] + -1.0*value[28]*other_value[23] + 1.0*value[18]*other_value[30] + 1.0*value[1]*other_value[13] + 1.0*value[5]*other_value[29]

@cuda.jit(device=True)
def gmt_func_o20(value, other_value):
    return 0 + -1.0*value[21]*other_value[13] + -1.0*value[3]*other_value[9] + -1.0*value[9]*other_value[3] + 1.0*value[1]*other_value[14] + 1.0*value[25]*other_value[8] + 1.0*value[0]*other_value[20] + 1.0*value[4]*other_value[29] + 1.0*value[24]*other_value[26] + 1.0*value[11]*other_value[31] + -1.0*value[2]*other_value[27] + 1.0*value[13]*other_value[21] + 1.0*value[12]*other_value[16] + 1.0*value[30]*other_value[17] + 1.0*value[7]*other_value[5] + -1.0*value[8]*other_value[25] + 1.0*value[6]*other_value[23] + 1.0*value[26]*other_value[24] + -1.0*value[16]*other_value[12] + -1.0*value[10]*other_value[18] + -1.0*value[15]*other_value[19] + 1.0*value[31]*other_value[11] + 1.0*value[20]*other_value[0] + 1.0*value[5]*other_value[7] + -1.0*value[28]*other_value[22] + 1.0*value[17]*other_value[30] + -1.0*value[22]*other_value[28] + 1.0*value[18]*other_value[10] + -1.0*value[29]*other_value[4] + 1.0*value[19]*other_value[15] + 1.0*value[14]*other_value[1] + -1.0*value[23]*other_value[6] + 1.0*value[27]*other_value[2]

@cuda.jit(device=True)
def gmt_func_o21(value, other_value):
    return 0 + -1.0*value[24]*other_value[6] + -1.0*value[11]*other_value[18] + -1.0*value[26]*other_value[23] + 1.0*value[18]*other_value[11] + -1.0*value[23]*other_value[26] + -1.0*value[9]*other_value[4] + -1.0*value[10]*other_value[31] + -1.0*value[30]*other_value[16] + 1.0*value[27]*other_value[22] + 1.0*value[5]*other_value[8] + 1.0*value[22]*other_value[27] + -1.0*value[2]*other_value[28] + -1.0*value[16]*other_value[30] + 1.0*value[8]*other_value[5] + 1.0*value[0]*other_value[21] + 1.0*value[20]*other_value[13] + 1.0*value[12]*other_value[17] + -1.0*value[19]*other_value[14] + 1.0*value[15]*other_value[1] + -1.0*value[3]*other_value[29] + 1.0*value[28]*other_value[2] + -1.0*value[25]*other_value[7] + 1.0*value[29]*other_value[3] + 1.0*value[7]*other_value[25] + 1.0*value[1]*other_value[15] + 1.0*value[21]*other_value[0] + -1.0*value[4]*other_value[9] + -1.0*value[31]*other_value[10] + 1.0*value[6]*other_value[24] + -1.0*value[17]*other_value[12] + 1.0*value[14]*other_value[19] + -1.0*value[13]*other_value[20]

@cuda.jit(device=True)
def gmt_func_o22(value, other_value):
    return 0 + 1.0*value[28]*other_value[20] + 1.0*value[4]*other_value[10] + 1.0*value[2]*other_value[13] + 1.0*value[10]*other_value[4] + -1.0*value[26]*other_value[1] + -1.0*value[3]*other_value[11] + -1.0*value[27]*other_value[21] + -1.0*value[31]*other_value[9] + 1.0*value[7]*other_value[17] + 1.0*value[25]*other_value[12] + -1.0*value[29]*other_value[18] + 1.0*value[5]*other_value[30] + -1.0*value[6]*other_value[19] + -1.0*value[8]*other_value[16] + -1.0*value[24]*other_value[14] + -1.0*value[30]*other_value[5] + -1.0*value[9]*other_value[31] + 1.0*value[23]*other_value[15] + 1.0*value[0]*other_value[22] + 1.0*value[16]*other_value[8] + -1.0*value[17]*other_value[7] + 1.0*value[20]*other_value[28] + -1.0*value[11]*other_value[3] + -1.0*value[12]*other_value[25] + 1.0*value[13]*other_value[2] + -1.0*value[18]*other_value[29] + 1.0*value[1]*other_value[26] + -1.0*value[15]*other_value[23] + 1.0*value[14]*other_value[24] + 1.0*value[19]*other_value[6] + -1.0*value[21]*other_value[27] + 1.0*value[22]*other_value[0]

@cuda.jit(device=True)
def gmt_func_o23(value, other_value):
    return 0 + 1.0*value[2]*other_value[14] + 1.0*value[4]*other_value[30] + -1.0*value[24]*other_value[13] + -1.0*value[8]*other_value[31] + 1.0*value[0]*other_value[23] + -1.0*value[11]*other_value[25] + 1.0*value[14]*other_value[2] + -1.0*value[21]*other_value[26] + -1.0*value[29]*other_value[17] + -1.0*value[18]*other_value[7] + 1.0*value[25]*other_value[11] + 1.0*value[7]*other_value[18] + 1.0*value[20]*other_value[6] + 1.0*value[19]*other_value[28] + -1.0*value[17]*other_value[29] + -1.0*value[6]*other_value[20] + -1.0*value[26]*other_value[21] + -1.0*value[31]*other_value[8] + -1.0*value[27]*other_value[1] + 1.0*value[13]*other_value[24] + 1.0*value[5]*other_value[10] + -1.0*value[12]*other_value[3] + 1.0*value[28]*other_value[19] + 1.0*value[22]*other_value[15] + 1.0*value[10]*other_value[5] + -1.0*value[30]*other_value[4] + 1.0*value[16]*other_value[9] + -1.0*value[3]*other_value[12] + 1.0*value[1]*other_value[27] + -1.0*value[15]*other_value[22] + -1.0*value[9]*other_value[16] + 1.0*value[23]*other_value[0]

@cuda.jit(device=True)
def gmt_func_o24(value, other_value):
    return 0 + -1.0*value[3]*other_value[30] + -1.0*value[4]*other_value[12] + 1.0*value[1]*other_value[28] + 1.0*value[7]*other_value[31] + -1.0*value[19]*other_value[27] + 1.0*value[0]*other_value[24] + 1.0*value[5]*other_value[11] + 1.0*value[10]*other_value[25] + 1.0*value[20]*other_value[26] + 1.0*value[14]*other_value[22] + 1.0*value[26]*other_value[20] + -1.0*value[12]*other_value[4] + 1.0*value[15]*other_value[2] + -1.0*value[25]*other_value[10] + -1.0*value[28]*other_value[1] + 1.0*value[29]*other_value[16] + 1.0*value[8]*other_value[18] + -1.0*value[6]*other_value[21] + 1.0*value[16]*other_value[29] + 1.0*value[30]*other_value[3] + 1.0*value[11]*other_value[5] + -1.0*value[13]*other_value[23] + -1.0*value[18]*other_value[8] + 1.0*value[24]*other_value[0] + 1.0*value[17]*other_value[9] + 1.0*value[2]*other_value[15] + 1.0*value[31]*other_value[7] + -1.0*value[27]*other_value[19] + 1.0*value[21]*other_value[6] + 1.0*value[23]*other_value[13] + -1.0*value[22]*other_value[14] + -1.0*value[9]*other_value[17]

@cuda.jit(device=True)
def gmt_func_o25(value, other_value):
    return 0 + -1.0*value[29]*other_value[1] + -1.0*value[20]*other_value[8] + -1.0*value[28]*other_value[16] + -1.0*value[16]*other_value[28] + 1.0*value[27]*other_value[17] + -1.0*value[12]*other_value[22] + -1.0*value[30]*other_value[2] + -1.0*value[7]*other_value[21] + -1.0*value[6]*other_value[31] + 1.0*value[5]*other_value[13] + 1.0*value[22]*other_value[12] + 1.0*value[17]*other_value[27] + -1.0*value[26]*other_value[18] + -1.0*value[18]*other_value[26] + 1.0*value[15]*other_value[3] + 1.0*value[3]*other_value[15] + 1.0*value[2]*other_value[30] + -1.0*value[31]*other_value[6] + 1.0*value[1]*other_value[29] + 1.0*value[19]*other_value[9] + 1.0*value[24]*other_value[10] + -1.0*value[4]*other_value[14] + 1.0*value[21]*other_value[7] + 1.0*value[8]*other_value[20] + -1.0*value[10]*other_value[24] + 1.0*value[11]*other_value[23] + 1.0*value[0]*other_value[25] + 1.0*value[13]*other_value[5] + -1.0*value[9]*other_value[19] + -1.0*value[23]*other_value[11] + -1.0*value[14]*other_value[4] + 1.0*value[25]*other_value[0]

@cuda.jit(device=True)
def gmt_func_o26(value, other_value):
    return 0 + 1.0*value[10]*other_value[8] + 1.0*value[13]*other_value[6] + -1.0*value[4]*other_value[16] + -1.0*value[5]*other_value[31] + -1.0*value[15]*other_value[27] + 1.0*value[6]*other_value[13] + 1.0*value[14]*other_value[28] + -1.0*value[23]*other_value[21] + -1.0*value[31]*other_value[5] + -1.0*value[12]*other_value[29] + -1.0*value[11]*other_value[7] + -1.0*value[21]*other_value[23] + 1.0*value[3]*other_value[17] + 1.0*value[0]*other_value[26] + -1.0*value[2]*other_value[19] + 1.0*value[24]*other_value[20] + -1.0*value[17]*other_value[3] + 1.0*value[29]*other_value[12] + 1.0*value[9]*other_value[30] + 1.0*value[19]*other_value[2] + -1.0*value[22]*other_value[1] + 1.0*value[8]*other_value[10] + -1.0*value[18]*other_value[25] + -1.0*value[30]*other_value[9] + -1.0*value[25]*other_value[18] + 1.0*value[16]*other_value[4] + -1.0*value[7]*other_value[11] + -1.0*value[28]*other_value[14] + 1.0*value[26]*other_value[0] + 1.0*value[1]*other_value[22] + 1.0*value[20]*other_value[24] + 1.0*value[27]*other_value[15]

@cuda.jit(device=True)
def gmt_func_o27(value, other_value):
    return 0 + 1.0*value[13]*other_value[28] + -1.0*value[15]*other_value[26] + -1.0*value[4]*other_value[31] + 1.0*value[14]*other_value[6] + -1.0*value[28]*other_value[13] + -1.0*value[30]*other_value[8] + -1.0*value[5]*other_value[16] + 1.0*value[19]*other_value[24] + 1.0*value[27]*other_value[0] + -1.0*value[22]*other_value[21] + 1.0*value[10]*other_value[9] + -1.0*value[23]*other_value[1] + 1.0*value[9]*other_value[10] + 1.0*value[8]*other_value[30] + 1.0*value[16]*other_value[5] + -1.0*value[21]*other_value[22] + 1.0*value[3]*other_value[18] + 1.0*value[1]*other_value[23] + 1.0*value[24]*other_value[19] + 1.0*value[29]*other_value[11] + -1.0*value[11]*other_value[29] + -1.0*value[31]*other_value[4] + -1.0*value[17]*other_value[25] + 1.0*value[20]*other_value[2] + -1.0*value[18]*other_value[3] + 1.0*value[6]*other_value[14] + -1.0*value[12]*other_value[7] + -1.0*value[25]*other_value[17] + -1.0*value[7]*other_value[12] + 1.0*value[0]*other_value[27] + -1.0*value[2]*other_value[20] + 1.0*value[26]*other_value[15]

@cuda.jit(device=True)
def gmt_func_o28(value, other_value):
    return 0 + 1.0*value[0]*other_value[28] + 1.0*value[14]*other_value[26] + 1.0*value[30]*other_value[7] + 1.0*value[4]*other_value[18] + 1.0*value[3]*other_value[31] + -1.0*value[29]*other_value[10] + -1.0*value[18]*other_value[4] + 1.0*value[17]*other_value[5] + 1.0*value[15]*other_value[6] + 1.0*value[31]*other_value[3] + -1.0*value[26]*other_value[14] + -1.0*value[7]*other_value[30] + 1.0*value[20]*other_value[22] + -1.0*value[12]*other_value[8] + 1.0*value[16]*other_value[25] + 1.0*value[25]*other_value[16] + 1.0*value[1]*other_value[24] + 1.0*value[11]*other_value[9] + -1.0*value[24]*other_value[1] + 1.0*value[9]*other_value[11] + -1.0*value[23]*other_value[19] + 1.0*value[21]*other_value[2] + -1.0*value[19]*other_value[23] + 1.0*value[6]*other_value[15] + -1.0*value[8]*other_value[12] + -1.0*value[2]*other_value[21] + 1.0*value[22]*other_value[20] + 1.0*value[28]*other_value[0] + 1.0*value[27]*other_value[13] + -1.0*value[5]*other_value[17] + -1.0*value[13]*other_value[27] + 1.0*value[10]*other_value[29]

@cuda.jit(device=True)
def gmt_func_o29(value, other_value):
    return 0 + 1.0*value[0]*other_value[29] + 1.0*value[15]*other_value[7] + 1.0*value[1]*other_value[25] + 1.0*value[9]*other_value[13] + -1.0*value[2]*other_value[31] + -1.0*value[24]*other_value[16] + -1.0*value[3]*other_value[21] + -1.0*value[8]*other_value[14] + 1.0*value[23]*other_value[17] + -1.0*value[30]*other_value[6] + -1.0*value[10]*other_value[28] + 1.0*value[28]*other_value[10] + 1.0*value[17]*other_value[23] + 1.0*value[11]*other_value[27] + -1.0*value[14]*other_value[8] + -1.0*value[18]*other_value[22] + 1.0*value[4]*other_value[20] + -1.0*value[5]*other_value[19] + -1.0*value[27]*other_value[11] + 1.0*value[19]*other_value[5] + 1.0*value[21]*other_value[3] + 1.0*value[26]*other_value[12] + -1.0*value[22]*other_value[18] + 1.0*value[6]*other_value[30] + -1.0*value[20]*other_value[4] + 1.0*value[7]*other_value[15] + 1.0*value[29]*other_value[0] + -1.0*value[31]*other_value[2] + -1.0*value[16]*other_value[24] + -1.0*value[25]*other_value[1] + -1.0*value[12]*other_value[26] + 1.0*value[13]*other_value[9]

@cuda.jit(device=True)
def gmt_func_o30(value, other_value):
    return 0 + 1.0*value[2]*other_value[25] + 1.0*value[16]*other_value[21] + 1.0*value[10]*other_value[15] + 1.0*value[0]*other_value[30] + 1.0*value[22]*other_value[5] + 1.0*value[9]*other_value[26] + -1.0*value[23]*other_value[4] + 1.0*value[1]*other_value[31] + -1.0*value[8]*other_value[27] + 1.0*value[24]*other_value[3] + -1.0*value[11]*other_value[14] + 1.0*value[21]*other_value[16] + -1.0*value[25]*other_value[2] + 1.0*value[7]*other_value[28] + 1.0*value[12]*other_value[13] + -1.0*value[20]*other_value[17] + 1.0*value[30]*other_value[0] + -1.0*value[6]*other_value[29] + 1.0*value[13]*other_value[12] + 1.0*value[19]*other_value[18] + 1.0*value[27]*other_value[8] + -1.0*value[5]*other_value[22] + 1.0*value[18]*other_value[19] + -1.0*value[14]*other_value[11] + -1.0*value[28]*other_value[7] + 1.0*value[4]*other_value[23] + 1.0*value[29]*other_value[6] + -1.0*value[17]*other_value[20] + -1.0*value[3]*other_value[24] + 1.0*value[15]*other_value[10] + -1.0*value[26]*other_value[9] + 1.0*value[31]*other_value[1]

@cuda.jit(device=True)
def gmt_func_o31(value, other_value):
    return 0 + -1.0*value[24]*other_value[7] + 1.0*value[1]*other_value[30] + -1.0*value[17]*other_value[14] + 1.0*value[15]*other_value[16] + 1.0*value[18]*other_value[13] + -1.0*value[14]*other_value[17] + 1.0*value[13]*other_value[18] + 1.0*value[19]*other_value[12] + -1.0*value[20]*other_value[11] + 1.0*value[12]*other_value[19] + -1.0*value[11]*other_value[20] + 1.0*value[21]*other_value[10] + 1.0*value[10]*other_value[21] + -1.0*value[22]*other_value[9] + 1.0*value[23]*other_value[8] + 1.0*value[31]*other_value[0] + -1.0*value[9]*other_value[22] + 1.0*value[8]*other_value[23] + -1.0*value[7]*other_value[24] + 1.0*value[25]*other_value[6] + 1.0*value[26]*other_value[5] + 1.0*value[6]*other_value[25] + -1.0*value[27]*other_value[4] + 1.0*value[5]*other_value[26] + 1.0*value[28]*other_value[3] + -1.0*value[4]*other_value[27] + -1.0*value[29]*other_value[2] + 1.0*value[3]*other_value[28] + 1.0*value[30]*other_value[1] + -1.0*value[2]*other_value[29] + 1.0*value[0]*other_value[31] + 1.0*value[16]*other_value[15]

@cuda.jit(device=True)
def gmt_func(value, other_value, output):
    output[0] = gmt_func_o0(value,other_value)
    output[1] = gmt_func_o1(value,other_value)
    output[2] = gmt_func_o2(value,other_value)
    output[3] = gmt_func_o3(value,other_value)
    output[4] = gmt_func_o4(value,other_value)
    output[5] = gmt_func_o5(value,other_value)
    output[6] = gmt_func_o6(value,other_value)
    output[7] = gmt_func_o7(value,other_value)
    output[8] = gmt_func_o8(value,other_value)
    output[9] = gmt_func_o9(value,other_value)
    output[10] = gmt_func_o10(value,other_value)
    output[11] = gmt_func_o11(value,other_value)
    output[12] = gmt_func_o12(value,other_value)
    output[13] = gmt_func_o13(value,other_value)
    output[14] = gmt_func_o14(value,other_value)
    output[15] = gmt_func_o15(value,other_value)
    output[16] = gmt_func_o16(value,other_value)
    output[17] = gmt_func_o17(value,other_value)
    output[18] = gmt_func_o18(value,other_value)
    output[19] = gmt_func_o19(value,other_value)
    output[20] = gmt_func_o20(value,other_value)
    output[21] = gmt_func_o21(value,other_value)
    output[22] = gmt_func_o22(value,other_value)
    output[23] = gmt_func_o23(value,other_value)
    output[24] = gmt_func_o24(value,other_value)
    output[25] = gmt_func_o25(value,other_value)
    output[26] = gmt_func_o26(value,other_value)
    output[27] = gmt_func_o27(value,other_value)
    output[28] = gmt_func_o28(value,other_value)
    output[29] = gmt_func_o29(value,other_value)
    output[30] = gmt_func_o30(value,other_value)
    output[31] = gmt_func_o31(value,other_value)

@cuda.jit(device=True)
def omt_func_o0(value, other_value):
    return 0 + 1.0*value[0]*other_value[0]

@cuda.jit(device=True)
def omt_func_o1(value, other_value):
    return 0 + 1.0*value[0]*other_value[1] + 1.0*value[1]*other_value[0]

@cuda.jit(device=True)
def omt_func_o2(value, other_value):
    return 0 + 1.0*value[0]*other_value[2] + 1.0*value[2]*other_value[0]

@cuda.jit(device=True)
def omt_func_o3(value, other_value):
    return 0 + 1.0*value[3]*other_value[0] + 1.0*value[0]*other_value[3]

@cuda.jit(device=True)
def omt_func_o4(value, other_value):
    return 0 + 1.0*value[4]*other_value[0] + 1.0*value[0]*other_value[4]

@cuda.jit(device=True)
def omt_func_o5(value, other_value):
    return 0 + 1.0*value[5]*other_value[0] + 1.0*value[0]*other_value[5]

@cuda.jit(device=True)
def omt_func_o6(value, other_value):
    return 0 + 1.0*value[1]*other_value[2] + 1.0*value[6]*other_value[0] + -1.0*value[2]*other_value[1] + 1.0*value[0]*other_value[6]

@cuda.jit(device=True)
def omt_func_o7(value, other_value):
    return 0 + 1.0*value[1]*other_value[3] + 1.0*value[0]*other_value[7] + -1.0*value[3]*other_value[1] + 1.0*value[7]*other_value[0]

@cuda.jit(device=True)
def omt_func_o8(value, other_value):
    return 0 + 1.0*value[0]*other_value[8] + 1.0*value[1]*other_value[4] + 1.0*value[8]*other_value[0] + -1.0*value[4]*other_value[1]

@cuda.jit(device=True)
def omt_func_o9(value, other_value):
    return 0 + 1.0*value[9]*other_value[0] + 1.0*value[0]*other_value[9] + 1.0*value[1]*other_value[5] + -1.0*value[5]*other_value[1]

@cuda.jit(device=True)
def omt_func_o10(value, other_value):
    return 0 + -1.0*value[3]*other_value[2] + 1.0*value[2]*other_value[3] + 1.0*value[0]*other_value[10] + 1.0*value[10]*other_value[0]

@cuda.jit(device=True)
def omt_func_o11(value, other_value):
    return 0 + -1.0*value[4]*other_value[2] + 1.0*value[2]*other_value[4] + 1.0*value[0]*other_value[11] + 1.0*value[11]*other_value[0]

@cuda.jit(device=True)
def omt_func_o12(value, other_value):
    return 0 + 1.0*value[2]*other_value[5] + 1.0*value[12]*other_value[0] + 1.0*value[0]*other_value[12] + -1.0*value[5]*other_value[2]

@cuda.jit(device=True)
def omt_func_o13(value, other_value):
    return 0 + 1.0*value[13]*other_value[0] + -1.0*value[4]*other_value[3] + 1.0*value[0]*other_value[13] + 1.0*value[3]*other_value[4]

@cuda.jit(device=True)
def omt_func_o14(value, other_value):
    return 0 + 1.0*value[3]*other_value[5] + -1.0*value[5]*other_value[3] + 1.0*value[14]*other_value[0] + 1.0*value[0]*other_value[14]

@cuda.jit(device=True)
def omt_func_o15(value, other_value):
    return 0 + 1.0*value[4]*other_value[5] + -1.0*value[5]*other_value[4] + 1.0*value[15]*other_value[0] + 1.0*value[0]*other_value[15]

@cuda.jit(device=True)
def omt_func_o16(value, other_value):
    return 0 + 1.0*value[10]*other_value[1] + 1.0*value[16]*other_value[0] + 1.0*value[3]*other_value[6] + 1.0*value[6]*other_value[3] + -1.0*value[7]*other_value[2] + 1.0*value[1]*other_value[10] + 1.0*value[0]*other_value[16] + -1.0*value[2]*other_value[7]

@cuda.jit(device=True)
def omt_func_o17(value, other_value):
    return 0 + 1.0*value[11]*other_value[1] + 1.0*value[17]*other_value[0] + 1.0*value[1]*other_value[11] + -1.0*value[8]*other_value[2] + 1.0*value[4]*other_value[6] + -1.0*value[2]*other_value[8] + 1.0*value[6]*other_value[4] + 1.0*value[0]*other_value[17]

@cuda.jit(device=True)
def omt_func_o18(value, other_value):
    return 0 + 1.0*value[1]*other_value[12] + 1.0*value[6]*other_value[5] + 1.0*value[5]*other_value[6] + 1.0*value[0]*other_value[18] + 1.0*value[18]*other_value[0] + 1.0*value[12]*other_value[1] + -1.0*value[2]*other_value[9] + -1.0*value[9]*other_value[2]

@cuda.jit(device=True)
def omt_func_o19(value, other_value):
    return 0 + 1.0*value[4]*other_value[7] + 1.0*value[1]*other_value[13] + 1.0*value[19]*other_value[0] + 1.0*value[7]*other_value[4] + 1.0*value[0]*other_value[19] + 1.0*value[13]*other_value[1] + -1.0*value[8]*other_value[3] + -1.0*value[3]*other_value[8]

@cuda.jit(device=True)
def omt_func_o20(value, other_value):
    return 0 + 1.0*value[7]*other_value[5] + 1.0*value[0]*other_value[20] + -1.0*value[9]*other_value[3] + 1.0*value[20]*other_value[0] + 1.0*value[14]*other_value[1] + 1.0*value[5]*other_value[7] + -1.0*value[3]*other_value[9] + 1.0*value[1]*other_value[14]

@cuda.jit(device=True)
def omt_func_o21(value, other_value):
    return 0 + 1.0*value[1]*other_value[15] + 1.0*value[15]*other_value[1] + -1.0*value[9]*other_value[4] + 1.0*value[21]*other_value[0] + 1.0*value[0]*other_value[21] + 1.0*value[5]*other_value[8] + -1.0*value[4]*other_value[9] + 1.0*value[8]*other_value[5]

@cuda.jit(device=True)
def omt_func_o22(value, other_value):
    return 0 + 1.0*value[13]*other_value[2] + 1.0*value[0]*other_value[22] + 1.0*value[10]*other_value[4] + -1.0*value[11]*other_value[3] + 1.0*value[2]*other_value[13] + -1.0*value[3]*other_value[11] + 1.0*value[22]*other_value[0] + 1.0*value[4]*other_value[10]

@cuda.jit(device=True)
def omt_func_o23(value, other_value):
    return 0 + 1.0*value[5]*other_value[10] + 1.0*value[2]*other_value[14] + 1.0*value[10]*other_value[5] + 1.0*value[0]*other_value[23] + -1.0*value[12]*other_value[3] + -1.0*value[3]*other_value[12] + 1.0*value[14]*other_value[2] + 1.0*value[23]*other_value[0]

@cuda.jit(device=True)
def omt_func_o24(value, other_value):
    return 0 + -1.0*value[12]*other_value[4] + 1.0*value[0]*other_value[24] + 1.0*value[24]*other_value[0] + 1.0*value[15]*other_value[2] + 1.0*value[2]*other_value[15] + 1.0*value[11]*other_value[5] + 1.0*value[5]*other_value[11] + -1.0*value[4]*other_value[12]

@cuda.jit(device=True)
def omt_func_o25(value, other_value):
    return 0 + 1.0*value[25]*other_value[0] + 1.0*value[3]*other_value[15] + 1.0*value[0]*other_value[25] + -1.0*value[4]*other_value[14] + 1.0*value[5]*other_value[13] + 1.0*value[15]*other_value[3] + -1.0*value[14]*other_value[4] + 1.0*value[13]*other_value[5]

@cuda.jit(device=True)
def omt_func_o26(value, other_value):
    return 0 + -1.0*value[17]*other_value[3] + 1.0*value[10]*other_value[8] + -1.0*value[2]*other_value[19] + -1.0*value[4]*other_value[16] + -1.0*value[11]*other_value[7] + 1.0*value[26]*other_value[0] + 1.0*value[3]*other_value[17] + 1.0*value[1]*other_value[22] + 1.0*value[0]*other_value[26] + -1.0*value[22]*other_value[1] + 1.0*value[8]*other_value[10] + 1.0*value[13]*other_value[6] + 1.0*value[19]*other_value[2] + 1.0*value[16]*other_value[4] + -1.0*value[7]*other_value[11] + 1.0*value[6]*other_value[13]

@cuda.jit(device=True)
def omt_func_o27(value, other_value):
    return 0 + -1.0*value[23]*other_value[1] + 1.0*value[14]*other_value[6] + -1.0*value[12]*other_value[7] + 1.0*value[27]*other_value[0] + -1.0*value[18]*other_value[3] + 1.0*value[20]*other_value[2] + 1.0*value[16]*other_value[5] + 1.0*value[10]*other_value[9] + 1.0*value[6]*other_value[14] + 1.0*value[0]*other_value[27] + 1.0*value[1]*other_value[23] + 1.0*value[9]*other_value[10] + -1.0*value[2]*other_value[20] + -1.0*value[5]*other_value[16] + 1.0*value[3]*other_value[18] + -1.0*value[7]*other_value[12]

@cuda.jit(device=True)
def omt_func_o28(value, other_value):
    return 0 + 1.0*value[15]*other_value[6] + -1.0*value[5]*other_value[17] + -1.0*value[24]*other_value[1] + 1.0*value[1]*other_value[24] + 1.0*value[0]*other_value[28] + 1.0*value[4]*other_value[18] + -1.0*value[18]*other_value[4] + 1.0*value[6]*other_value[15] + 1.0*value[21]*other_value[2] + 1.0*value[17]*other_value[5] + 1.0*value[9]*other_value[11] + 1.0*value[11]*other_value[9] + -1.0*value[8]*other_value[12] + -1.0*value[2]*other_value[21] + 1.0*value[28]*other_value[0] + -1.0*value[12]*other_value[8]

@cuda.jit(device=True)
def omt_func_o29(value, other_value):
    return 0 + 1.0*value[19]*other_value[5] + -1.0*value[3]*other_value[21] + -1.0*value[20]*other_value[4] + 1.0*value[0]*other_value[29] + 1.0*value[29]*other_value[0] + 1.0*value[21]*other_value[3] + 1.0*value[15]*other_value[7] + 1.0*value[9]*other_value[13] + 1.0*value[13]*other_value[9] + -1.0*value[5]*other_value[19] + -1.0*value[8]*other_value[14] + 1.0*value[1]*other_value[25] + -1.0*value[14]*other_value[8] + -1.0*value[25]*other_value[1] + 1.0*value[7]*other_value[15] + 1.0*value[4]*other_value[20]

@cuda.jit(device=True)
def omt_func_o30(value, other_value):
    return 0 + 1.0*value[22]*other_value[5] + -1.0*value[25]*other_value[2] + -1.0*value[23]*other_value[4] + -1.0*value[3]*other_value[24] + 1.0*value[24]*other_value[3] + 1.0*value[30]*other_value[0] + 1.0*value[2]*other_value[25] + 1.0*value[10]*other_value[15] + 1.0*value[12]*other_value[13] + 1.0*value[4]*other_value[23] + -1.0*value[11]*other_value[14] + -1.0*value[14]*other_value[11] + -1.0*value[5]*other_value[22] + 1.0*value[0]*other_value[30] + 1.0*value[13]*other_value[12] + 1.0*value[15]*other_value[10]

@cuda.jit(device=True)
def omt_func_o31(value, other_value):
    return 0 + -1.0*value[24]*other_value[7] + 1.0*value[30]*other_value[1] + -1.0*value[7]*other_value[24] + 1.0*value[12]*other_value[19] + 1.0*value[18]*other_value[13] + 1.0*value[26]*other_value[5] + 1.0*value[8]*other_value[23] + -1.0*value[27]*other_value[4] + -1.0*value[11]*other_value[20] + 1.0*value[28]*other_value[3] + -1.0*value[29]*other_value[2] + -1.0*value[9]*other_value[22] + 1.0*value[25]*other_value[6] + 1.0*value[13]*other_value[18] + 1.0*value[6]*other_value[25] + 1.0*value[23]*other_value[8] + -1.0*value[17]*other_value[14] + 1.0*value[3]*other_value[28] + 1.0*value[10]*other_value[21] + 1.0*value[19]*other_value[12] + -1.0*value[4]*other_value[27] + 1.0*value[0]*other_value[31] + -1.0*value[2]*other_value[29] + 1.0*value[16]*other_value[15] + 1.0*value[15]*other_value[16] + 1.0*value[21]*other_value[10] + 1.0*value[5]*other_value[26] + -1.0*value[14]*other_value[17] + -1.0*value[22]*other_value[9] + 1.0*value[1]*other_value[30] + -1.0*value[20]*other_value[11] + 1.0*value[31]*other_value[0]

@cuda.jit(device=True)
def omt_func(value, other_value, output):
    output[0] = omt_func_o0(value,other_value)
    output[1] = omt_func_o1(value,other_value)
    output[2] = omt_func_o2(value,other_value)
    output[3] = omt_func_o3(value,other_value)
    output[4] = omt_func_o4(value,other_value)
    output[5] = omt_func_o5(value,other_value)
    output[6] = omt_func_o6(value,other_value)
    output[7] = omt_func_o7(value,other_value)
    output[8] = omt_func_o8(value,other_value)
    output[9] = omt_func_o9(value,other_value)
    output[10] = omt_func_o10(value,other_value)
    output[11] = omt_func_o11(value,other_value)
    output[12] = omt_func_o12(value,other_value)
    output[13] = omt_func_o13(value,other_value)
    output[14] = omt_func_o14(value,other_value)
    output[15] = omt_func_o15(value,other_value)
    output[16] = omt_func_o16(value,other_value)
    output[17] = omt_func_o17(value,other_value)
    output[18] = omt_func_o18(value,other_value)
    output[19] = omt_func_o19(value,other_value)
    output[20] = omt_func_o20(value,other_value)
    output[21] = omt_func_o21(value,other_value)
    output[22] = omt_func_o22(value,other_value)
    output[23] = omt_func_o23(value,other_value)
    output[24] = omt_func_o24(value,other_value)
    output[25] = omt_func_o25(value,other_value)
    output[26] = omt_func_o26(value,other_value)
    output[27] = omt_func_o27(value,other_value)
    output[28] = omt_func_o28(value,other_value)
    output[29] = omt_func_o29(value,other_value)
    output[30] = omt_func_o30(value,other_value)
    output[31] = omt_func_o31(value,other_value)

@cuda.jit(device=True)
def imt_func_o0(value, other_value):
    return 0 + 1.0*value[1]*other_value[1] + -1.0*value[5]*other_value[5] + -1.0*value[6]*other_value[6] + -1.0*value[7]*other_value[7] + -1.0*value[8]*other_value[8] + 1.0*value[9]*other_value[9] + -1.0*value[10]*other_value[10] + -1.0*value[11]*other_value[11] + 1.0*value[12]*other_value[12] + -1.0*value[13]*other_value[13] + 1.0*value[14]*other_value[14] + 1.0*value[15]*other_value[15] + -1.0*value[16]*other_value[16] + 1.0*value[4]*other_value[4] + -1.0*value[17]*other_value[17] + -1.0*value[19]*other_value[19] + 1.0*value[20]*other_value[20] + 1.0*value[21]*other_value[21] + -1.0*value[22]*other_value[22] + 1.0*value[23]*other_value[23] + 1.0*value[24]*other_value[24] + 1.0*value[25]*other_value[25] + 1.0*value[26]*other_value[26] + -1.0*value[27]*other_value[27] + -1.0*value[28]*other_value[28] + -1.0*value[29]*other_value[29] + -1.0*value[30]*other_value[30] + 1.0*value[18]*other_value[18] + 1.0*value[3]*other_value[3] + -1.0*value[31]*other_value[31] + 1.0*value[2]*other_value[2]

@cuda.jit(device=True)
def imt_func_o1(value, other_value):
    return 0 + -1.0*value[17]*other_value[11] + 1.0*value[5]*other_value[9] + 1.0*value[22]*other_value[26] + -1.0*value[26]*other_value[22] + 1.0*value[6]*other_value[2] + 1.0*value[29]*other_value[25] + 1.0*value[7]*other_value[3] + 1.0*value[21]*other_value[15] + 1.0*value[8]*other_value[4] + 1.0*value[20]*other_value[14] + -1.0*value[9]*other_value[5] + -1.0*value[19]*other_value[13] + -1.0*value[10]*other_value[16] + 1.0*value[28]*other_value[24] + -1.0*value[11]*other_value[17] + 1.0*value[12]*other_value[18] + 1.0*value[18]*other_value[12] + -1.0*value[13]*other_value[19] + -1.0*value[2]*other_value[6] + 1.0*value[14]*other_value[20] + 1.0*value[27]*other_value[23] + -1.0*value[31]*other_value[30] + 1.0*value[15]*other_value[21] + -1.0*value[30]*other_value[31] + -1.0*value[23]*other_value[27] + -1.0*value[16]*other_value[10] + -1.0*value[3]*other_value[7] + -1.0*value[24]*other_value[28] + -1.0*value[25]*other_value[29] + -1.0*value[4]*other_value[8]

@cuda.jit(device=True)
def imt_func_o2(value, other_value):
    return 0 + 1.0*value[10]*other_value[3] + 1.0*value[24]*other_value[15] + 1.0*value[7]*other_value[16] + 1.0*value[21]*other_value[28] + 1.0*value[8]*other_value[17] + 1.0*value[16]*other_value[7] + -1.0*value[9]*other_value[18] + -1.0*value[19]*other_value[26] + -1.0*value[28]*other_value[21] + -1.0*value[12]*other_value[5] + -1.0*value[25]*other_value[30] + -1.0*value[13]*other_value[22] + -1.0*value[18]*other_value[9] + -1.0*value[27]*other_value[20] + 1.0*value[14]*other_value[23] + -1.0*value[3]*other_value[10] + 1.0*value[31]*other_value[29] + 1.0*value[15]*other_value[24] + 1.0*value[17]*other_value[8] + 1.0*value[11]*other_value[4] + 1.0*value[29]*other_value[31] + 1.0*value[20]*other_value[27] + 1.0*value[1]*other_value[6] + 1.0*value[5]*other_value[12] + -1.0*value[6]*other_value[1] + 1.0*value[23]*other_value[14] + 1.0*value[26]*other_value[19] + -1.0*value[22]*other_value[13] + -1.0*value[4]*other_value[11] + 1.0*value[30]*other_value[25]

@cuda.jit(device=True)
def imt_func_o3(value, other_value):
    return 0 + 1.0*value[27]*other_value[18] + -1.0*value[18]*other_value[27] + -1.0*value[12]*other_value[23] + -1.0*value[6]*other_value[16] + 1.0*value[19]*other_value[8] + -1.0*value[10]*other_value[2] + -1.0*value[28]*other_value[31] + 1.0*value[25]*other_value[15] + 1.0*value[5]*other_value[14] + 1.0*value[2]*other_value[10] + -1.0*value[9]*other_value[20] + -1.0*value[30]*other_value[24] + -1.0*value[14]*other_value[5] + 1.0*value[17]*other_value[26] + -1.0*value[31]*other_value[28] + -1.0*value[23]*other_value[12] + 1.0*value[15]*other_value[25] + -1.0*value[26]*other_value[17] + -1.0*value[16]*other_value[6] + 1.0*value[13]*other_value[4] + -1.0*value[4]*other_value[13] + 1.0*value[11]*other_value[22] + 1.0*value[8]*other_value[19] + 1.0*value[24]*other_value[30] + -1.0*value[20]*other_value[9] + 1.0*value[1]*other_value[7] + -1.0*value[29]*other_value[21] + -1.0*value[7]*other_value[1] + 1.0*value[21]*other_value[29] + 1.0*value[22]*other_value[11]

@cuda.jit(device=True)
def imt_func_o4(value, other_value):
    return 0 + -1.0*value[8]*other_value[1] + -1.0*value[12]*other_value[24] + -1.0*value[18]*other_value[28] + -1.0*value[13]*other_value[3] + -1.0*value[21]*other_value[9] + -1.0*value[24]*other_value[12] + -1.0*value[7]*other_value[19] + -1.0*value[14]*other_value[25] + -1.0*value[22]*other_value[10] + 1.0*value[5]*other_value[15] + -1.0*value[15]*other_value[5] + 1.0*value[29]*other_value[20] + -1.0*value[17]*other_value[6] + -1.0*value[16]*other_value[26] + 1.0*value[27]*other_value[31] + -1.0*value[6]*other_value[17] + 1.0*value[3]*other_value[13] + 1.0*value[2]*other_value[11] + 1.0*value[26]*other_value[16] + -1.0*value[20]*other_value[29] + -1.0*value[25]*other_value[14] + -1.0*value[10]*other_value[22] + -1.0*value[11]*other_value[2] + 1.0*value[31]*other_value[27] + 1.0*value[30]*other_value[23] + -1.0*value[23]*other_value[30] + 1.0*value[1]*other_value[8] + -1.0*value[19]*other_value[7] + -1.0*value[9]*other_value[21] + 1.0*value[28]*other_value[18]

@cuda.jit(device=True)
def imt_func_o5(value, other_value):
    return 0 + 1.0*value[4]*other_value[15] + 1.0*value[1]*other_value[9] + -1.0*value[16]*other_value[27] + 1.0*value[31]*other_value[26] + -1.0*value[19]*other_value[29] + -1.0*value[15]*other_value[4] + -1.0*value[24]*other_value[11] + 1.0*value[29]*other_value[19] + -1.0*value[17]*other_value[28] + -1.0*value[10]*other_value[23] + -1.0*value[9]*other_value[1] + -1.0*value[6]*other_value[18] + -1.0*value[12]*other_value[2] + 1.0*value[3]*other_value[14] + -1.0*value[14]*other_value[3] + -1.0*value[23]*other_value[10] + 1.0*value[27]*other_value[16] + 1.0*value[30]*other_value[22] + -1.0*value[11]*other_value[24] + -1.0*value[21]*other_value[8] + -1.0*value[8]*other_value[21] + 1.0*value[28]*other_value[17] + -1.0*value[20]*other_value[7] + 1.0*value[26]*other_value[31] + -1.0*value[13]*other_value[25] + -1.0*value[22]*other_value[30] + -1.0*value[18]*other_value[6] + -1.0*value[7]*other_value[20] + 1.0*value[2]*other_value[12] + -1.0*value[25]*other_value[13]

@cuda.jit(device=True)
def imt_func_o6(value, other_value):
    return 0 + -1.0*value[26]*other_value[13] + 1.0*value[25]*other_value[31] + 1.0*value[17]*other_value[4] + 1.0*value[27]*other_value[14] + -1.0*value[18]*other_value[5] + -1.0*value[13]*other_value[26] + 1.0*value[3]*other_value[16] + 1.0*value[31]*other_value[25] + 1.0*value[15]*other_value[28] + 1.0*value[16]*other_value[3] + 1.0*value[28]*other_value[15] + 1.0*value[14]*other_value[27] + 1.0*value[4]*other_value[17] + -1.0*value[5]*other_value[18]

@cuda.jit(device=True)
def imt_func_o7(value, other_value):
    return 0 + -1.0*value[16]*other_value[2] + -1.0*value[20]*other_value[5] + 1.0*value[19]*other_value[4] + 1.0*value[11]*other_value[26] + -1.0*value[12]*other_value[27] + 1.0*value[29]*other_value[15] + -1.0*value[5]*other_value[20] + -1.0*value[2]*other_value[16] + 1.0*value[4]*other_value[19] + -1.0*value[24]*other_value[31] + -1.0*value[31]*other_value[24] + 1.0*value[15]*other_value[29] + 1.0*value[26]*other_value[11] + -1.0*value[27]*other_value[12]

@cuda.jit(device=True)
def imt_func_o8(value, other_value):
    return 0 + -1.0*value[5]*other_value[21] + 1.0*value[31]*other_value[23] + -1.0*value[21]*other_value[5] + -1.0*value[2]*other_value[17] + -1.0*value[26]*other_value[10] + -1.0*value[3]*other_value[19] + 1.0*value[23]*other_value[31] + -1.0*value[29]*other_value[14] + -1.0*value[17]*other_value[2] + -1.0*value[28]*other_value[12] + -1.0*value[12]*other_value[28] + -1.0*value[19]*other_value[3] + -1.0*value[14]*other_value[29] + -1.0*value[10]*other_value[26]

@cuda.jit(device=True)
def imt_func_o9(value, other_value):
    return 0 + -1.0*value[2]*other_value[18] + -1.0*value[11]*other_value[28] + -1.0*value[21]*other_value[4] + 1.0*value[31]*other_value[22] + 1.0*value[22]*other_value[31] + -1.0*value[13]*other_value[29] + -1.0*value[29]*other_value[13] + -1.0*value[28]*other_value[11] + -1.0*value[10]*other_value[27] + -1.0*value[20]*other_value[3] + -1.0*value[27]*other_value[10] + -1.0*value[4]*other_value[21] + -1.0*value[18]*other_value[2] + -1.0*value[3]*other_value[20]

@cuda.jit(device=True)
def imt_func_o10(value, other_value):
    return 0 + 1.0*value[30]*other_value[15] + 1.0*value[31]*other_value[21] + -1.0*value[23]*other_value[5] + -1.0*value[5]*other_value[23] + 1.0*value[22]*other_value[4] + 1.0*value[4]*other_value[22] + 1.0*value[1]*other_value[16] + 1.0*value[21]*other_value[31] + 1.0*value[15]*other_value[30] + -1.0*value[8]*other_value[26] + 1.0*value[27]*other_value[9] + 1.0*value[16]*other_value[1] + -1.0*value[26]*other_value[8] + 1.0*value[9]*other_value[27]

@cuda.jit(device=True)
def imt_func_o11(value, other_value):
    return 0 + -1.0*value[14]*other_value[30] + -1.0*value[3]*other_value[22] + 1.0*value[1]*other_value[17] + -1.0*value[24]*other_value[5] + 1.0*value[28]*other_value[9] + 1.0*value[17]*other_value[1] + 1.0*value[9]*other_value[28] + -1.0*value[31]*other_value[20] + -1.0*value[20]*other_value[31] + 1.0*value[7]*other_value[26] + -1.0*value[30]*other_value[14] + -1.0*value[5]*other_value[24] + 1.0*value[26]*other_value[7] + -1.0*value[22]*other_value[3]

@cuda.jit(device=True)
def imt_func_o12(value, other_value):
    return 0 + 1.0*value[18]*other_value[1] + -1.0*value[19]*other_value[31] + 1.0*value[28]*other_value[8] + 1.0*value[8]*other_value[28] + -1.0*value[31]*other_value[19] + 1.0*value[27]*other_value[7] + -1.0*value[3]*other_value[23] + -1.0*value[23]*other_value[3] + 1.0*value[7]*other_value[27] + -1.0*value[4]*other_value[24] + -1.0*value[30]*other_value[13] + -1.0*value[24]*other_value[4] + -1.0*value[13]*other_value[30] + 1.0*value[1]*other_value[18]

@cuda.jit(device=True)
def imt_func_o13(value, other_value):
    return 0 + 1.0*value[30]*other_value[12] + 1.0*value[29]*other_value[9] + 1.0*value[31]*other_value[18] + -1.0*value[26]*other_value[6] + 1.0*value[19]*other_value[1] + 1.0*value[18]*other_value[31] + -1.0*value[6]*other_value[26] + -1.0*value[5]*other_value[25] + 1.0*value[2]*other_value[22] + 1.0*value[22]*other_value[2] + 1.0*value[1]*other_value[19] + 1.0*value[12]*other_value[30] + -1.0*value[25]*other_value[5] + 1.0*value[9]*other_value[29]

@cuda.jit(device=True)
def imt_func_o14(value, other_value):
    return 0 + -1.0*value[27]*other_value[6] + 1.0*value[17]*other_value[31] + 1.0*value[29]*other_value[8] + 1.0*value[2]*other_value[23] + 1.0*value[20]*other_value[1] + -1.0*value[6]*other_value[27] + 1.0*value[11]*other_value[30] + 1.0*value[30]*other_value[11] + 1.0*value[31]*other_value[17] + 1.0*value[8]*other_value[29] + -1.0*value[4]*other_value[25] + -1.0*value[25]*other_value[4] + 1.0*value[1]*other_value[20] + 1.0*value[23]*other_value[2]

@cuda.jit(device=True)
def imt_func_o15(value, other_value):
    return 0 + -1.0*value[28]*other_value[6] + -1.0*value[30]*other_value[10] + -1.0*value[6]*other_value[28] + -1.0*value[10]*other_value[30] + -1.0*value[7]*other_value[29] + -1.0*value[29]*other_value[7] + -1.0*value[16]*other_value[31] + 1.0*value[2]*other_value[24] + 1.0*value[3]*other_value[25] + 1.0*value[25]*other_value[3] + -1.0*value[31]*other_value[16] + 1.0*value[21]*other_value[1] + 1.0*value[24]*other_value[2] + 1.0*value[1]*other_value[21]

@cuda.jit(device=True)
def imt_func_o16(value, other_value):
    return 0 + 1.0*value[26]*other_value[4] + -1.0*value[27]*other_value[5] + 1.0*value[5]*other_value[27] + 1.0*value[15]*other_value[31] + 1.0*value[31]*other_value[15] + -1.0*value[4]*other_value[26]

@cuda.jit(device=True)
def imt_func_o17(value, other_value):
    return 0 + 1.0*value[3]*other_value[26] + -1.0*value[31]*other_value[14] + 1.0*value[5]*other_value[28] + -1.0*value[26]*other_value[3] + -1.0*value[28]*other_value[5] + -1.0*value[14]*other_value[31]

@cuda.jit(device=True)
def imt_func_o18(value, other_value):
    return 0 + -1.0*value[31]*other_value[13] + -1.0*value[27]*other_value[3] + 1.0*value[4]*other_value[28] + 1.0*value[3]*other_value[27] + -1.0*value[28]*other_value[4] + -1.0*value[13]*other_value[31]

@cuda.jit(device=True)
def imt_func_o19(value, other_value):
    return 0 + -1.0*value[29]*other_value[5] + 1.0*value[31]*other_value[12] + -1.0*value[2]*other_value[26] + 1.0*value[5]*other_value[29] + 1.0*value[26]*other_value[2] + 1.0*value[12]*other_value[31]

@cuda.jit(device=True)
def imt_func_o20(value, other_value):
    return 0 + -1.0*value[29]*other_value[4] + 1.0*value[4]*other_value[29] + 1.0*value[27]*other_value[2] + 1.0*value[31]*other_value[11] + -1.0*value[2]*other_value[27] + 1.0*value[11]*other_value[31]

@cuda.jit(device=True)
def imt_func_o21(value, other_value):
    return 0 + 1.0*value[28]*other_value[2] + -1.0*value[10]*other_value[31] + -1.0*value[2]*other_value[28] + -1.0*value[3]*other_value[29] + -1.0*value[31]*other_value[10] + 1.0*value[29]*other_value[3]

@cuda.jit(device=True)
def imt_func_o22(value, other_value):
    return 0 + 1.0*value[5]*other_value[30] + -1.0*value[30]*other_value[5] + -1.0*value[31]*other_value[9] + -1.0*value[26]*other_value[1] + 1.0*value[1]*other_value[26] + -1.0*value[9]*other_value[31]

@cuda.jit(device=True)
def imt_func_o23(value, other_value):
    return 0 + -1.0*value[31]*other_value[8] + 1.0*value[4]*other_value[30] + -1.0*value[27]*other_value[1] + -1.0*value[30]*other_value[4] + -1.0*value[8]*other_value[31] + 1.0*value[1]*other_value[27]

@cuda.jit(device=True)
def imt_func_o24(value, other_value):
    return 0 + 1.0*value[31]*other_value[7] + 1.0*value[7]*other_value[31] + 1.0*value[1]*other_value[28] + -1.0*value[3]*other_value[30] + -1.0*value[28]*other_value[1] + 1.0*value[30]*other_value[3]

@cuda.jit(device=True)
def imt_func_o25(value, other_value):
    return 0 + 1.0*value[2]*other_value[30] + 1.0*value[1]*other_value[29] + -1.0*value[30]*other_value[2] + -1.0*value[6]*other_value[31] + -1.0*value[29]*other_value[1] + -1.0*value[31]*other_value[6]

@cuda.jit(device=True)
def imt_func_o26(value, other_value):
    return 0 + -1.0*value[31]*other_value[5] + -1.0*value[5]*other_value[31]

@cuda.jit(device=True)
def imt_func_o27(value, other_value):
    return 0 + -1.0*value[31]*other_value[4] + -1.0*value[4]*other_value[31]

@cuda.jit(device=True)
def imt_func_o28(value, other_value):
    return 0 + 1.0*value[31]*other_value[3] + 1.0*value[3]*other_value[31]

@cuda.jit(device=True)
def imt_func_o29(value, other_value):
    return 0 + -1.0*value[31]*other_value[2] + -1.0*value[2]*other_value[31]

@cuda.jit(device=True)
def imt_func_o30(value, other_value):
    return 0 + 1.0*value[1]*other_value[31] + 1.0*value[31]*other_value[1]

@cuda.jit(device=True)
def imt_func_o31(value, other_value):
    return 0

@cuda.jit(device=True)
def imt_func(value, other_value, output):
    output[0] = imt_func_o0(value,other_value)
    output[1] = imt_func_o1(value,other_value)
    output[2] = imt_func_o2(value,other_value)
    output[3] = imt_func_o3(value,other_value)
    output[4] = imt_func_o4(value,other_value)
    output[5] = imt_func_o5(value,other_value)
    output[6] = imt_func_o6(value,other_value)
    output[7] = imt_func_o7(value,other_value)
    output[8] = imt_func_o8(value,other_value)
    output[9] = imt_func_o9(value,other_value)
    output[10] = imt_func_o10(value,other_value)
    output[11] = imt_func_o11(value,other_value)
    output[12] = imt_func_o12(value,other_value)
    output[13] = imt_func_o13(value,other_value)
    output[14] = imt_func_o14(value,other_value)
    output[15] = imt_func_o15(value,other_value)
    output[16] = imt_func_o16(value,other_value)
    output[17] = imt_func_o17(value,other_value)
    output[18] = imt_func_o18(value,other_value)
    output[19] = imt_func_o19(value,other_value)
    output[20] = imt_func_o20(value,other_value)
    output[21] = imt_func_o21(value,other_value)
    output[22] = imt_func_o22(value,other_value)
    output[23] = imt_func_o23(value,other_value)
    output[24] = imt_func_o24(value,other_value)
    output[25] = imt_func_o25(value,other_value)
    output[26] = imt_func_o26(value,other_value)
    output[27] = imt_func_o27(value,other_value)
    output[28] = imt_func_o28(value,other_value)
    output[29] = imt_func_o29(value,other_value)
    output[30] = imt_func_o30(value,other_value)
    output[31] = imt_func_o31(value,other_value)

