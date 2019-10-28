from . import Cl, conformalize
layout_orig, blades_orig = Cl(3)
layout, blades, stuff = conformalize(layout_orig)


locals().update(blades)
locals().update(stuff)
