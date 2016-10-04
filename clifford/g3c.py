from . import Cl
layout, blades = Cl(3)
layout, blades, stuff = conformalize(layout)

locals().update(blades)
locals().update(stuff)
