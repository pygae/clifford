from . import Cl

layout, blades = Cl(3, 0, 1, firstIdx=0)
locals().update(blades)

# for shorter reprs
layout.__name__ = 'layout'
layout.__module__ = __name__
