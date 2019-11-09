"""
Tools to generate G3 objects randomly
"""

import numpy as np
import numba

from . import generate_rotation_rotor, np_to_euc_mv


def random_unit_vector():
    """ Creates a random unit vector """
    return (np_to_euc_mv(np.random.randn(3))).normal()


@numba.njit
def val_random_euc_mv(l_max=10):
    """ Creates a random vector normally distributed with length l_max """
    output = np.zeros(32)
    np_in = l_max*np.random.randn(3)
    output[1] = np_in[0]
    output[2] = np_in[1]
    output[3] = np_in[2]
    return output


def random_euc_mv(l_max=10):
    """ Creates a random vector normally distributed with length l_max """
    return np_to_euc_mv(l_max*np.random.randn(3))


def random_rotation_rotor(max_angle=np.pi):
    """ Creates a random rotation rotor """
    return generate_rotation_rotor(max_angle * np.random.rand(), random_unit_vector(), random_unit_vector())
