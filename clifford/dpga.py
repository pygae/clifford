""" 
The Cl(4,4) DPGA also known as `Mother Algebra' of Goldman and Mann. see:

R(4, 4) As a Computational Framework for 3-Dimensional Computer Graphics
Ron Goldman and Stephen Mann

and

Modeling 3D Geometry in the Clifford Algebra R(4, 4)
Juan Du, Ron Goldman & Stephen Mann

and

Transverse approach to geometric algebra models for manipulating quadratic surfaces
Stéphane Breuils, Vincent Nozick, Laurent Fuchs, Akihiro Sugimoto
June 2019 Calgary Canada
"""

from . import Layout, BasisVectorIds
import numpy as np


layout = Layout([1]*4 + [-1]*4, ids=BasisVectorIds(['{}'.format(i) for i in range(4)] + ['{}b'.format(i) for i in range(4)]))
blades = layout.bases()
locals().update(blades)


# for shorter reprs
layout.__name__ = 'layout'
layout.__module__ = __name__

# Construct the non diagonal basis

w1 = 0.5*(e1 + e1b)
w2 = 0.5*(e2 + e2b)
w3 = 0.5*(e3 + e3b)
w0 = 0.5*(e0 + e0b)
wlist = [w1, w2, w3, w0]

w1s = 0.5*(e1 - e1b)
w2s = 0.5*(e2 - e2b)
w3s = 0.5*(e3 - e3b)
w0s = 0.5*(e0 - e0b)
wslist = [w1s, w2s, w3s, w0s]

wbasis = wlist + wslist


def up(threedDvec):
    x, y, z = threedDvec
    return x*w1 + y*w2 + z*w3 + w0


def down(pnt):
    return np.array([(pnt|wis)[()] for wis in [w1s, w2s, w3s]])/((pnt|w0s)[()])


def dual_point(point):
    x, y, z = down(point)
    return x * w1s + y * w2s + z * w3s + w0s
