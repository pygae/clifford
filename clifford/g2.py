from . import Cl
layout, blades = Cl(2)
locals().update(blades)

# for shorter reprs
layout.__name__ = 'layout'
layout.__module__ = __name__
