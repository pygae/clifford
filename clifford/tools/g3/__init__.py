
from clifford.g3c import *
import clifford as cf
import math
import numpy as np


def quaternion_to_rotor(quaternion):
    """
    Converts a quaternion into a pure rotation rotor
    """
    quat_W = quaternion[0]
    quat_X = quaternion[1]
    quat_Y = quaternion[2]
    quat_Z = quaternion[3]
    i = e23
    j = e13
    k = e12
    return quat_W + quat_X * i + quat_Y * j + quat_Z * k


def rotor_to_quaternion(rotor):
    """
    Converts a pure rotation rotor into a quaternion
    """
    quat = np.zeros(4)
    quat[0] = rotor[0]
    quat[1] = rotor[(2, 3)]
    quat[2] = rotor[(1, 3)]
    quat[3] = rotor[(1, 2)]
    return quat


def quaternion_to_matrix(q):
    """
    Converts a quaternion into a rotation matrix
    """
    qz2 = 2*q[3]**2
    qy2 = 2*q[2]**2
    qx2 = 2*q[1]**2
    qxy = 2*q[1]*q[2]
    qxz = 2*q[1]*q[3]
    qxw = 2*q[1]*q[0]
    qyw = 2*q[2]*q[0]
    qyz = 2*q[2]*q[3]
    qzw = 2*q[3]*q[0]
    return np.array([[1-qy2-qz2, qxy-qzw, qxz+qyw],[qxy+qzw , 1-qx2-qz2, qyz-qxw],[qxz-qyw, qyz+qxw, 1-qx2-qy2]])


def rotor_to_rotation_matrix(R):
    """
    Converts a pure rotation rotor into a rotation matrix
    """
    q = rotor_to_quaternion(R)
    return quaternion_to_matrix(q)




