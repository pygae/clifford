
import numpy as np
import clifford as cf
import numba
from clifford.g3c import *
from . import mult_with_ninf

ninf = einf
no = -eo
I5 = e12345
I3 = e123

unit_scalar_mv = 1.0 + 0.0*e1
imt_func = layout.imt_func
gmt_func = layout.gmt_func
adjoint_func = layout.adjoint_func
e4_val = e4.value
ninf_val = einf.value
I5_val = I5.value
no_val = no.value
I3_val = I3.value


@numba.njit
def val_exp(B_val):
    """
    Fast implementation of the exp function - JITTED
    """
    t_val = imt_func(B_val, no_val)

    phiP_val = B_val - mult_with_ninf(t_val)
    phi = np.sqrt(-float(gmt_func(phiP_val, phiP_val)[0]))
    P_val = phiP_val / phi

    P_n_val = gmt_func(P_val, I3_val)
    t_nor_val = gmt_func(imt_func(t_val, P_n_val), P_n_val)
    t_par_val = t_val - t_nor_val

    coef_val = np.sin(phi) * P_val
    coef_val[0] += np.cos(phi)

    R_val = coef_val + gmt_func(coef_val, mult_with_ninf(t_nor_val)) + \
        np.sinc(phi/np.pi) * mult_with_ninf(t_par_val)
    return R_val


def ga_exp(B):
    """
    Fast implementation of the exp function
    """
    if np.sum(np.abs(B.value)) < np.finfo(float).eps:
        return cf.MultiVector(layout, unit_scalar_mv.value)
    return cf.MultiVector(layout, val_exp(B.value))


def interpolate_rotors(R_n_plus_1, R_n, interpolation_fraction):
    """
    Mesh Vertex Pose and Position Interpolation using Geometric Algebra.
    Rich Wareham and Joan Lasenby
    """
    if interpolation_fraction < np.finfo(float).eps:
        return R_n
    delta_R = R_n_plus_1 * ~R_n
    delta_bivector = ga_log(delta_R)
    R_n_lambda = ga_exp(interpolation_fraction * delta_bivector) * R_n
    return R_n_lambda


def extractRotorComponents(R):
    phi = np.arccos(float(R[0]))             #scalar
    phi2 = phi * phi                  #scalar
    #Notice: np.sinc(pi * x)/(pi x)
    phi_sinc = np.sinc(phi/np.pi)             #scalar
    phiP = ((R(2)*ninf)|ep)/(phi_sinc)
    t_normal_n = -((phiP * R(4))/(phi2 * phi_sinc))
    t_perpendicular_n = -(phiP * (phiP * R(2))(2))/(phi2 * phi_sinc)
    return phiP, t_normal_n, t_perpendicular_n


def ga_log(R):
    """
    R must be a displacement rotor. grades in [0, 2, 4]

    Presented by R. Wareham (Applications of CGA)


    WARNING: DOES NOT COMMUTE log(A * B) != log(A) + log(B)
    """
    phiP, t_normal_n, t_perpendicular_n = extractRotorComponents(R)
    return phiP + t_normal_n + t_perpendicular_n


@numba.njit
def val_vec_repr_to_bivector(x):
    """
    Converts between the parameters of a bivector and the bivector itself
    """
    t_val = np.zeros(32)
    t_val[1] = x[0]
    t_val[2] = x[1]
    t_val[3] = x[2]
    B_val = gmt_func(t_val, ninf_val)
    B_val[6] += x[3]
    B_val[7] += x[4]
    B_val[10] += x[5]
    return B_val


@numba.njit
def val_rotorconversion(x):
    """
    Converts between the parameters of a bivector and the rotor that it is generating
    """
    B_val = val_vec_repr_to_bivector(x)
    R_val = val_exp(B_val)
    return R_val


def rotorconversion(x):
    """
    Converts between the parameters of a bivector and the rotor that it is generating
    """
    return cf.MultiVector(layout, val_rotorconversion(x))
