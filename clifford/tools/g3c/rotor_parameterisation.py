
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


def general_exp(x, order=9):
    """
    This implements the series expansion of e**mv where mv is a multivector
    The parameter order is the maximum order of the taylor series to use
    """

    result = 1.0
    if (order == 0):
        return result

    # scale by power of 2 so that its norm is < 1
    max_val = int(np.max(np.abs(x.value)))
    scale=1
    if max_val > 1:
        max_val <<= 1
    while max_val:
        max_val >>= 1
        scale <<= 1

    scaled = x * (1.0 / scale)

    # taylor approximation
    tmp = 1.0
    for i in range(1, order):
        tmp = tmp*scaled * (1.0 / i)
        result += tmp

    # undo scaling
    while scale > 1:
        result *= result
        scale >>= 1
    return result


def general_logarithm(V):
    """
    This implements the logarithm of a TRS rotor to a bivector
    Ie. any translation rotation and scaling rotor
    """
    epsilon = 10**(-6)
    R = V(e123).normal()
    RV = R*V

    # Extract the scaling
    tanh_gamma_2 = -RV[e45]/RV[0]
    gamma = 2*np.arctanh(tanh_gamma_2)

    if abs(gamma) < epsilon:
        gamma_dash = 1
    else:
        gamma_dash = gamma/(np.exp(gamma)-1)
    S = (np.cosh(gamma/2) + np.sinh(gamma/2)*(ninf^no)).normal()
    R = V(e123).normal()
    T = (V*~S*~R).normal()
    t = -2*(eo.lc(T))(1)
    if abs(R[0] - 1) < epsilon:  # No rotation
        biv_log = -gamma_dash*t*einf/2 + gamma*(eo^einf)/2
        return biv_log
    elif abs(gamma) < epsilon:  # No scaling component
            phi = np.arccos(float(V[0]))  # scalar
            phi2 = phi * phi  # scalar
            # Notice: np.sinc(pi * x)/(pi x)
            phi_sinc = np.sinc(phi / np.pi)  # scalar
            phiP = ((V(2) * ninf) | ep) / (phi_sinc)
            t_normal_n = -((phiP * V(4)) / (phi2 * phi_sinc))
            t_perpendicular_n = -(phiP * (phiP * V(2))(2)) / (phi2 * phi_sinc)
            return phiP + t_normal_n + t_perpendicular_n
    else:  # Definitely have rotation and scaling
        I = -R(2) / abs(R(2))
        phi = 2 * np.arctan2(abs(R(2)), R[0])
        if abs(t) > epsilon:  # Translation too, full triple whammy
            tIoverI = (t.lc(I) * ~I)
            A = 1 - (np.exp(gamma)) * (R * R)
            A_rev = ~A
            A_inv = A_rev/((A*A_rev)[0])
            Tv = (1 + ninf * A_inv * tIoverI*0.5 ).normal()
            t_perp = ((t ^ I) * ~I)
            biv_log_perp = -gamma_dash * t_perp * einf/2
            biv_log_par = Tv * (-I * phi / 2 - gamma * (no ^ ninf)/2 ) * ~Tv
            biv_log = biv_log_perp + biv_log_par
            return biv_log
        else:  # No translation, just rotation and scaling
            biv_log = -I * phi/2 - gamma * (no ^ ninf) / 2
            return biv_log


@numba.njit
def val_exp(B_val):
    """
    Fast implementation of the translation and rotation specific exp function
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
    Fast implementation of the translation and rotation specific exp function
    """
    if np.sum(np.abs(B.value)) < np.finfo(float).eps:
        return cf.MultiVector(layout, unit_scalar_mv.value)
    return cf.MultiVector(layout, val_exp(B.value))


def interpolate_TR_rotors(R_n_plus_1, R_n, interpolation_fraction):
    """
    Interpolates TR type rotors
    Mesh Vertex Pose and Position Interpolation using Geometric Algebra.
    Rich Wareham and Joan Lasenby
    """
    if interpolation_fraction < np.finfo(float).eps:
        return R_n
    delta_R = R_n_plus_1 * ~R_n
    delta_bivector = ga_log(delta_R)(2)
    R_n_lambda = ga_exp(interpolation_fraction * delta_bivector) * R_n
    return R_n_lambda


def interpolate_TRS_rotors(R_n_plus_1, R_n, interpolation_fraction):
    """
    Interpolates TR type rotors
    Leo Dorst GA for Computer Science
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
    Mesh Vertex Pose and Position Interpolation using Geometric Algebra.
    Rich Wareham and Joan Lasenby
    """
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
    R must be a TR rotor. grades in [0, 2, 4]

    Presented by R. Wareham (Applications of CGA)


    WARNING: DOES NOT COMMUTE log(A * B) != log(A) + log(B)
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
    B_val = gmt_func(t_val, ninf_val)
    B_val[6] += x[3]
    B_val[7] += x[4]
    B_val[10] += x[5]
    return B_val


@numba.njit
def val_rotorconversion(x):
    """
    Converts between the parameters of a TR bivector and the rotor that it is generating
    """
    B_val = val_vec_repr_to_bivector(x)
    R_val = val_exp(B_val)
    return R_val


def rotorconversion(x):
    """
    Converts between the parameters of a TR bivector and the rotor that it is generating
    """
    return cf.MultiVector(layout, val_rotorconversion(x))
