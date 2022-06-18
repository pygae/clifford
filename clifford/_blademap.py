from .transformations import LinearMatrix

class BladeMap:
    '''
    A Map Relating Blades in two different algebras.

    This is now just a thin wrapper around :func:`LinearMatrix.from_mapping`,
    which is more powerful.

    Examples
    -----------
    >>> from clifford import Cl
    >>> # Dirac Algebra  `D`
    >>> D, D_blades = Cl(1, 3, firstIdx=0, names='d')
    >>> locals().update(D_blades)

    >>> # Pauli Algebra  `P`
    >>> P, P_blades = Cl(3, names='p')
    >>> locals().update(P_blades)
    >>> sta_split = BladeMap([
    ...     (d01, p1),
    ...     (d02, p2),
    ...     (d03, p3),
    ...     (d12, p12),
    ...     (d23, p23),
    ...     (d13, p13),
    ...     (d0123, p123)
    ... ])
    '''
    def __init__(self, blades_map, map_scalars=True):
        self.blades_map = blades_map

        if map_scalars:
            # make scalars in each algebra map
            s1 = self.b1[0]._newMV(dtype=int)+1
            s2 = self.b2[0]._newMV(dtype=int)+1
            self.blades_map = [(s1, s2)] + self.blades_map

        self._transformation = LinearMatrix.from_mapping(self.blades_map)

    @property
    def b1(self):
        return [k[0] for k in self.blades_map]

    @property
    def b2(self):
        return [k[1] for k in self.blades_map]

    @property
    def layout1(self):
        return self._transformation.layout_src

    @property
    def layout2(self):
        return self._transformation.layout_dst

    def __call__(self, A):
        '''map an MV `A` according to blade_map'''

        # determine direction of map
        if A.layout == self._transformation.layout_src:
            return _transformation(A)
        elif A.layout == self.layout2:
            return _transformation.adjoint(A)
        else:
            raise ValueError('A doesnt belong to either Algebra in this Map')
