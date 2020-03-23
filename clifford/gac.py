from . import Cl
layout, blades = Cl(5, 3)
locals().update(blades)

# for shorter reprs
layout.__name__ = 'layout'
layout.__module__ = __name__

n1 = e3 + e6
n2 = e4 + e7
n3 = e5 + e8

n1b = 0.5*(e6 - e3)
n2b = 0.5*(e7 - e4)
n3b = 0.5*(e8 - e5)


def up(x):
    a = x[e1]
    b = x[e2]
    return a*e1 + b*e2 + 0.5*a**2*n1 + n1b + 0.5*b**2*n2 + n2b + a*b*n3


def down(x):
    return (x|e1)[0]*e1 + (x|e2)[0]*e2
