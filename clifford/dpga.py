
""" 
The Cl(4,4) DPGA of Goldman and Mann. see:

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

from . import Cl
import numpy as np

layout, blades = Cl(4, 4)
locals().update(blades)

# for shorter reprs
layout.__name__ = 'layout'
layout.__module__ = __name__

# Construct the non diagonal basis
e1b = e5
e2b = e6
e3b = e7
e4b = e8

w1 = 0.5*(e1 + e1b)
w2 = 0.5*(e2 + e2b)
w3 = 0.5*(e3 + e3b)
w0 = 0.5*(e4 + e4b)
wlist = [w1, w2, w3, w0]

w1s = 0.5*(e1 - e1b)
w2s = 0.5*(e2 - e2b)
w3s = 0.5*(e3 - e3b)
w0s = 0.5*(e4 - e4b)
wslist = [w1s, w2s, w3s, w0s]

wbasis = wlist + wslist


def up(threedDvec):
    x = threedDvec[0]
    y = threedDvec[1]
    z = threedDvec[2]
    return x*w1 + y*w2 + z*w3 + w0


def down(pnt):
    return np.array([(pnt|wis)[0] for wis in [w1s, w2s, w3s]])/((pnt|w0s)[0])
