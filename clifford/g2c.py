from . import Cl, conformalize
layout_orig, blades = Cl(2)
layout, blades, stuff = conformalize(layout_orig)


locals().update(blades)
locals().update(stuff)
