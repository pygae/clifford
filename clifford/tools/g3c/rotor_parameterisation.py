
import numpy as np
from scipy.optimize import minimize
import numba
from clifford.g3c import *
from . import mult_with_ninf, val_normalised
from clifford.tools.g3 import np_to_euc_mv
from clifford import general_exp

ninf = einf
no = -eo
I5 = e12345
I3 = e123

unit_scalar_mv = 1.0 + 0.0*e1
imt_func = layout.imt_func
gmt_func = layout.gmt_func
adjoint_func = layout.adjoint_func


def dorst_sinh(A):
    """
    sinh of a bivector as given in square root and logarithm of rotors
    by Dorst and Valkenburg
    """
    A2 = (A * A)[()]
    if A2 > 0:
        root_A2 = np.sqrt(A2)
        return (np.sinh(root_A2) / root_A2) * A
    elif abs(A2) < 0.000001:
        return +A
    else:
        root_A2 = np.sqrt(-A2)
        return (np.sin(root_A2) / root_A2) * A


def dorst_atanh2(s, c):
    """
    Atanh2 of a bivector as given in :cite:`log-of-rotors`
    """
    s2 = (s * s)[()]
    if s2 > 0:
        root_s2 = np.sqrt(s2)
        return (np.arcsinh(root_s2) / root_s2) * s
    elif abs(s2) < 0.000001:
        return +s
    else:
        root_s2 = np.sqrt(-s2)
        return (np.arctan2(root_s2, c) / root_s2) * s


def decompose_bivector(F):
    """
    Takes a bivector and decomposes it into 2 commuting blades
    From Hesternes and Sobzyk GA to GC p81 and with corrections
    by Anthony Lasenby
    """
    c1 = F
    F2 = F * F
    if F2 == 0:
        return +F, 0*e1
    c2 = 0.5 * F2(4)
    c1_2 = (c1 * c1)[()]
    c2_2 = (c2 * c2)[()]
    lambs = np.roots([1, -c1_2, c2_2])
    F1 = (c1 * c2 - lambs[0] * c1) / (lambs[1] - lambs[0])
    F2 = (c1 * c2 - lambs[1] * c1) / (lambs[0] - lambs[1])
    return F1, F2


def general_logarithm(R):
    """
    Takes a general conformal rotor and returns the log

    From :cite:`log-of-rotors`.
    """
    F = 2 * (R(4) - R[()]) * R(2)
    S1, S2 = decompose_bivector(F)

    def dorst_cosh(S):
        s2 = (S * S)[()]
        if abs(s2) < 0.000001:
            return (R ** 2)[()]
        else:
            return -((R ** 2)(2) * (S / s2))[()]

    C1 = dorst_cosh(S2)
    C2 = dorst_cosh(S1)

    return -0.5 * (dorst_atanh2(S1, C1) + dorst_atanh2(S2, C2))


def full_conformal_biv_params_to_biv(biv_params):
    """
    Converts the bivector parameters for a general conformal rotor into
    the bivector itself
    """
    phiP = np_to_euc_mv(biv_params[0:3]) * e123
    t = np_to_euc_mv(biv_params[3:6])
    s = np_to_euc_mv(biv_params[6:9])
    omega = biv_params[9]
    biv = phiP + t * ninf + s * no + omega * e45
    return biv


def full_conformal_biv_params_to_rotor(biv_params):
    """
    Converts the bivector parameters for a general conformal rotor into
    the rotor
    """
    biv = full_conformal_biv_params_to_biv(biv_params)
    R = general_exp(biv).normal()
    return R


def TRS_biv_params_to_biv(biv_params):
    """
    Converts the bivector parameters for a general TRS rotor into
    the bivector itself
    """
    phiP = np_to_euc_mv(biv_params[0:3]) * e123
    t = np_to_euc_mv(biv_params[3:6])
    omega = biv_params[6]
    biv = phiP + t * ninf + omega * e45
    return biv


def TRS_biv_params_to_rotor(biv_params):
    """
    Converts the bivector parameters for a general TRS rotor into
    the rotor
    """
    biv = TRS_biv_params_to_biv(biv_params)
    R = general_exp(biv).normal()
    return R


def find_closest_TRS_to_multivector(V):
    """
    Finds the closest TRS versor to the given multivector
    Distance is measured as l2 norm of coefficients
    """
    def residual_cost(biv_params):
        R = TRS_biv_params_to_biv(biv_params)
        return np.sum(np.abs(R.value - V.value)**2)
    x0 = np.random.randn(7) * 0.00001
    res = minimize(residual_cost, x0, method='L-BFGS-B')
    return TRS_biv_params_to_rotor(res.x).clean(0.00001).normal()


def find_closest_versor_to_multivector(V):
    """
    Finds the closest TRS versor to the given multivector
    Distance is measured as l2 norm of coefficients
    """
    def residual_cost(biv_params):
        R = full_conformal_biv_params_to_rotor(biv_params)
        return np.sum(np.abs(R.value - V.value)**2)
    x0 = np.random.randn(10) * 0.00001
    res = minimize(residual_cost, x0, method='L-BFGS-B')
    return TRS_biv_params_to_rotor(res.x).clean(0.00001).normal()


@numba.njit
def val_exp(B_val):
    """
    Fast implementation of the translation and rotation specific exp function
    """
    t_val = imt_func(B_val, no.value)

    phiP_val = B_val - mult_with_ninf(t_val)
    phi = np.sqrt(-float(gmt_func(phiP_val, phiP_val)[0]))
    P_val = phiP_val / phi

    P_n_val = gmt_func(P_val, I3.value)
    t_nor_val = gmt_func(imt_func(t_val, P_n_val), P_n_val)
    t_par_val = t_val - t_nor_val

    coef_val = np.sin(phi) * P_val
    coef_val[0] += np.cos(phi)

    R_val = coef_val + gmt_func(coef_val, mult_with_ninf(t_nor_val)) + \
        np.sinc(phi/np.pi) * mult_with_ninf(t_par_val)
    return R_val


def ga_exp(B):
    """
    Fast implementation of the translation and rotation specific exp function
    """
    if np.sum(np.abs(B.value)) < np.finfo(float).eps:
        return layout.MultiVector(unit_scalar_mv.value)
    return layout.MultiVector(val_exp(B.value))


def interpolate_TR_rotors(R_n_plus_1, R_n, interpolation_fraction):
    """
    Interpolates TR type rotors

    From :cite:`wareham-interpolation`.
    """
    if interpolation_fraction < np.finfo(float).eps:
        return R_n
    delta_R = R_n_plus_1 * ~R_n
    delta_bivector = ga_log(delta_R)(2)
    R_n_lambda = ga_exp(interpolation_fraction * delta_bivector) * R_n
    return R_n_lambda


def interpolate_rotors(R_n_plus_1, R_n, interpolation_fraction):
    """
    Interpolates all conformal type rotors
    From :cite:`wareham-interpolation` and :cite:`log-of-rotors`.
    """
    if interpolation_fraction < np.finfo(float).eps:
        return R_n
    delta_R = R_n_plus_1 * ~R_n
    delta_bivector = general_logarithm(delta_R)(2)
    R_n_lambda = general_exp(interpolation_fraction * delta_bivector) * R_n
    return R_n_lambda


def extractRotorComponents(R):
    """
    Extracts the translation and rotation information from a TR rotor

    From :cite:`wareham-interpolation`.
    """
    phi = np.arccos(R[()])             # scalar
    phi2 = phi * phi                  # scalar
    # Notice: np.sinc(pi * x)/(pi x)
    phi_sinc = np.sinc(phi/np.pi)             # scalar
    phiP = ((R(2)*ninf)|ep)/(phi_sinc)
    t_normal_n = -((phiP * R(4))/(phi2 * phi_sinc))
    t_perpendicular_n = -(phiP * (phiP * R(2))(2))/(phi2 * phi_sinc)
    return phiP, t_normal_n, t_perpendicular_n


def ga_log(R):
    """
    R must be a TR rotor. grades in [0, 2, 4]

    Presented in :cite:`wareham-applications`.

    .. warning::
        Does not commute, ``log(A * B) != log(A) + log(B)``
    """
    phiP, t_normal_n, t_perpendicular_n = extractRotorComponents(R)
    return phiP + t_normal_n + t_perpendicular_n


@numba.njit
def val_vec_repr_to_bivector(x):
    """
    Converts between the parameters of a TR bivector and the bivector itself
    """
    t_val = np.zeros(32)
    t_val[1] = x[0]
    t_val[2] = x[1]
    t_val[3] = x[2]
    B_val = gmt_func(t_val, ninf.value)
    B_val[6] += x[3]
    B_val[7] += x[4]
    B_val[10] += x[5]
    return B_val


val_TR_biv_params_to_biv = val_vec_repr_to_bivector


@numba.njit
def val_TR_biv_params_to_rotor(x):
    """
    Converts between the parameters of a TR bivector and the rotor that it is generating
    """
    B_val = val_vec_repr_to_bivector(x)
    R_val = val_exp(B_val)
    return R_val


val_rotorconversion = val_TR_biv_params_to_rotor


def TR_biv_params_to_rotor(x):
    """
    Converts between the parameters of a TR bivector and the rotor that it is generating
    """
    return layout.MultiVector(val_TR_biv_params_to_rotor(x))


rotorconversion = TR_biv_params_to_rotor


@numba.njit
def val_R_biv_params_to_rotor(x):
    """
    Converts between the parameters of an R only bivector and the rotor that it is generating
    """
    R_val = np.zeros(32)
    R_val[6] = x[0]
    R_val[7] += x[1]
    R_val[10] += x[2]
    R_val = val_exp(R_val)
    return R_val


def R_biv_params_to_rotor(x):
    """
    Converts between the parameters of a R bivector and the rotor that it is generating
    """
    return layout.MultiVector(val_R_biv_params_to_rotor(x))
