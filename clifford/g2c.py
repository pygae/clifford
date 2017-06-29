from . import Cl, conformalize
layout, blades = Cl(2)
layout, blades, stuff = conformalize(layout)

locals().update(blades)
locals().update(stuff)
