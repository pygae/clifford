"""
Tools to generate G3C objects randomly
"""

import numpy as np

from clifford.tools.g3 import generate_rotation_rotor
from clifford.tools.g3.random import (
    random_euc_mv, random_rotation_rotor, val_random_euc_mv,
)
from clifford.g3c import *

# TODO: resolve circular dependencies
from . import (
    ninf, val_normalised, omt_func, val_up, circle_to_sphere,
    generate_translation_rotor, I3
)


__all__ = [
    "random_point_pair_at_origin",
    "random_point_pair",
    "random_line_at_origin",
    "random_line",
    "random_circle_at_origin",
    "random_circle",
    "random_sphere_at_origin",
    "random_sphere",
    "random_plane_at_origin",
    "random_plane",
    "random_translation_rotor",
    "random_rotation_translation_rotor",
    "random_conformal_point",
    "random_bivector",
]


def random_point_pair_at_origin():
    """
    Creates a random point pair bivector object at the origin
    """
    mv_a = random_euc_mv()
    plane_a = (mv_a*I3).normal()

    mv_b = plane_a*mv_a*plane_a
    pp = (up(mv_a) ^ up(mv_b)).normal()
    return pp


def random_point_pair():
    """
    Creates a random point pair bivector object
    """
    mv_a = random_euc_mv()
    mv_b = random_euc_mv()
    pp = (up(mv_a) ^ up(mv_b)).normal()
    return pp


def random_line_at_origin():
    """
    Creates a random line at the origin
    """
    pp = (random_point_pair_at_origin()^einf).normal()
    return pp


def random_line():
    """
    Creates a random line
    """
    mv_a = random_euc_mv()
    mv_b = random_euc_mv()
    line_a = ((up(mv_a) ^ up(mv_b) ^ ninf)).normal()
    return line_a


def random_circle_at_origin():
    """
    Creates a random circle at the origin
    """
    mv_a = random_euc_mv()
    mv_r = random_euc_mv()
    r = generate_rotation_rotor(np.pi/2, mv_a, mv_r)
    mv_b = r*mv_a*~r
    mv_c = r * mv_b * ~r
    pp = (up(mv_a) ^ up(mv_b) ^ up(mv_c)).normal()
    return pp


def random_circle():
    """
    Creates a random circle
    """
    mv_a = val_random_euc_mv()
    mv_b = val_random_euc_mv()
    mv_c = val_random_euc_mv()
    return layout.MultiVector(value=val_normalised(omt_func(omt_func(val_up(mv_a), val_up(mv_b)), val_up(mv_c))))


def random_sphere_at_origin():
    """
    Creates a random sphere at the origin
    """
    C = random_circle_at_origin()
    sphere = circle_to_sphere(C)
    return sphere


def random_sphere():
    """
    Creates a random sphere
    """
    mv_a = random_euc_mv()
    mv_b = random_euc_mv()
    mv_c = random_euc_mv()
    mv_d = random_euc_mv()
    sphere = ((up(mv_a) ^ up(mv_b) ^ up(mv_c) ^ up(mv_d))).normal()
    return sphere


def random_plane_at_origin():
    """
    Creates a random plane at the origin
    """
    c = random_circle_at_origin()
    plane = (c ^ einf).normal()
    return plane


def random_plane():
    """
    Creates a random plane
    """
    c = random_circle()
    plane = (c ^ ninf).normal()
    return plane


def random_translation_rotor(maximum_translation=10.0):
    """ generate a random translation rotor """
    return generate_translation_rotor(random_euc_mv(maximum_translation))


def random_rotation_translation_rotor(maximum_translation=10.0, maximum_angle=np.pi):
    """ generate a random combined rotation and translation rotor """
    return (random_translation_rotor(maximum_translation)*random_rotation_rotor(maximum_angle)).normal()


def random_conformal_point(l_max=10):
    """
    Creates a random conformal point
    """
    return up(random_euc_mv(l_max=l_max))


def random_bivector():
    r"""
    Creates a random bivector on the form described by R. Wareham in
    Mesh Vertex Pose and Position Interpolation using Geometric Algebra.
    $$ B =  ab + c*n_{\inf}$$ where $a, b, c \in \mathcal(R)^3$
    """
    a = random_euc_mv()
    c = random_euc_mv()
    return a * I3 + c * ninf
