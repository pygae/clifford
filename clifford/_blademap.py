class BladeMap(object):
    '''
    A Map Relating Blades in two different algebras

    Examples
    -----------
    >>> from clifford import Cl
    >>> # Dirac Algebra  `D`
    >>> D, D_blades = Cl(1, 3, firstIdx=0, names='d')
    >>> locals().update(D_blades)

    >>> # Pauli Algebra  `P`
    >>> P, P_blades = Cl(3, names='p')
    >>> locals().update(P_blades)
    >>> sta_split = BladeMap([(d01, p1),
    ...                       (d02, p2),
    ...                       (d03, p3),
    ...                       (d12, p12),
    ...                       (d23, p23),
    ...                       (d13, p13)])

    '''
    def __init__(self, blades_map, map_scalars=True):
        self.blades_map = blades_map

        if map_scalars:
            # make scalars in each algebra map
            s1 = self.b1[0]._newMV(dtype=int)+1
            s2 = self.b2[0]._newMV(dtype=int)+1
            self.blades_map = [(s1, s2)] + self.blades_map

    @property
    def b1(self):
        return [k[0] for k in self.blades_map]

    @property
    def b2(self):
        return [k[1] for k in self.blades_map]

    @property
    def layout1(self):
        return self.b1[0].layout

    @property
    def layout2(self):
        return self.b2[0].layout

    def __call__(self, A):
        '''map an MV `A` according to blade_map'''

        # determine direction of map
        if A.layout == self.layout1:
            from_b = self.b1
            to_b = self.b2

        elif A.layout == self.layout2:
            from_b = self.b2
            to_b = self.b1
        else:
            raise ValueError('A doesnt belong to either Algebra in this Map')

        # create empty MV, and map values
        B = to_b[0]._newMV(dtype=int)
        for from_obj, to_obj in zip(from_b, to_b):
            B += (sum(A.value*from_obj.value)*to_obj)
        return B
