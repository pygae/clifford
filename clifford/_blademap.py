from typing import TypeVar

from ._layout import Layout
from ._multivector import MultiVector
from ._bit_helpers import BitPermuter


_SelfT = TypeVar('_SelfT', bound='_BaseTransformation')

class _BaseTransformation:
    """ Base class for transforming multivectors between one layout and another """
    def __init__(self, src_layout: Layout, dst_layout: Layout = None):
        self.src_layout = src_layout
        if dst_layout is None:
            dst_layout = src_layout
        self.dst_layout = dst_layout

    def __repr__(self):
        return "<{} from {} to {}>".format(type(self).__name__, self.src_layout, self.dst_layout)

    @property
    def inverse(self: _SelfT) -> _SelfT:
        """ The inverse of this mapping """
        self_rev = type(self).__new__()
        self_rev.__inverse_init__(self)
        return self

    def __inverse_init__(self: _SelfT, orig: _SelfT) -> None:
        """ A hook to be overriden by subclasses """
        self.src_layout = orig.dst_layout
        self.dst_layout = orig.src_layout


class _IndexTransformation(_BaseTransformation):
    """ A transformation that moves around multivector component indices """
    def __init__(self, src_layout: Layout, dst_layout: Layout, src_inds, dst_inds):
        super().__init__(src_layout, dst_layout)
        self._src_inds = src_inds
        self._dst_inds = dst_inds

    def __call__(self, mv_src: MultiVector) -> MultiVector:
        """ Apply this transformation to a MultiVector """
        if mv_src.layout != self.src_layout:
            raise ValueError("Multivector must belong to {}".format(self.src_layout))

        if self._dst_inds == slice(None):
            # optimization to avoid a copy
            return self.dst_layout.MultiVector(mv_src.value[self._src_inds])
        else:
            mv_dst = self.dst_layout.MultiVector(dtype=mv_src.value.dtype)
            mv_dst.value[self._dst_inds] = mv_src.value[self._src_inds]
            return mv_dst

    def __inverse_init__(self, orig):
        super().__inverse_init__(orig)
        self._dst_inds = orig._src_inds
        self._src_inds = orig._dst_inds


class BasisVectorMap(_IndexTransformation):
    """
    A map that converts blades from one algebra to another via converting the
    basis vectors.

    Note that does not support any scale factors, so is most useful for adding
    and removing basis vectors from multivectors, rather than for transforming
    the basis.

    Blades from the source algebra containing basis vectors not present in the
    destination algebra are dropped completely.
    """
    def __init__(
        self,
        src_layout: Layout, dst_layout: Layout,
        src_vectors: list = None, dst_vectors: list = None
    ):
        all_src_vectors = src_layout._basis_vector_ids.values
        all_dst_vectors = dst_layout._basis_vector_ids.values

        # handle default arguments
        if dst_vectors is None and src_vectors is None:
            # use the common vectors
            src_vectors = list(set(all_src_vectors) & set(all_dst_vectors))
            dst_vectors = src_vectors
        elif dst_vectors is None:
            dst_vectors = src_vectors

        try:
            src_bits = [all_src_vectors.index(f) for f in src_vectors]
        except ValueError:
            raise ValueError("Unknown ids {!r} in src_vectors".format(set(src_vectors) - set(all_src_vectors))) from None
        try:
            dst_bits = [all_dst_vectors.index(f) for f in dst_vectors]
        except ValueError:
            raise ValueError("Unknown ids {!r} in dst_vectors".format(set(dst_vectors) - set(all_dst_vectors))) from None

        # work out where each bit in `src` goes
        src_bit_to_dst_bit = {}
        for d, s in zip(dst_bits, src_bits):
            if s in src_bit_to_dst_bit:
                raise ValueError(
                    "Bit src[{}] maps to both dst[{}] and dst[{}]".format(s, src_bit_to_dst_bit[s], d))
            src_bit_to_dst_bit[s] = d

        permute = BitPermuter(src_bit_to_dst_bit)

        if len(src_bit_to_dst_bit) == len(all_dst_vectors):
            # every destination element has a source (some source bits may be discarded)
            src_inds = src_layout._basis_blade_order.bitmap_to_index[
                permute.inverse(dst_layout._basis_blade_order.index_to_bitmap)
            ]
            dst_inds = slice(None)
        elif len(src_bit_to_dst_bit) == len(all_src_vectors):
            # every source bit maps to a destination (some destination bits will be 0)
            src_inds = slice(None)
            dst_inds = dst_layout._basis_blade_order.bitmap_to_index[
                permute(src_layout._basis_blade_order.index_to_bitmap)
            ]
        else:
            # a combination of the above two cases
            src_inds = src_layout._basis_blade_order.index_to_bitmap & ~permute.domain == 0
            dst_inds = dst_layout._basis_blade_order.bitmap_to_index[
                permute.inverse(src_layout._basis_blade_order.index_to_bitmap[src_inds])
            ]

        super().__init__(src_layout, dst_layout, src_inds, dst_inds)
        self._src_vectors = src_vectors
        self._dst_vectors = dst_vectors

    def __inverse_init__(self, orig):
        super().__inverse_init__(orig)
        self._dst_vectors = orig._src_vectors
        self._src_vectors = orig._dst_vectors

    def __repr__(self):
        return "<{} from {} to {} with {}>".format(
            type(self).__name__,
            self.src_layout,
            self.dst_layout,
            ', '.join(
                '{}->{}'.format(s, d)
                for s, d in zip(self._src_vectors, self._dst_vectors)
            )
        )


class BladeMap(_BaseTransformation):
    '''
    A Map Relating Blades in two different algebras

    Examples
    --------
    >>> from clifford import Cl
    >>> # Dirac Algebra  `D`
    >>> D, D_blades = Cl(1, 3, firstIdx=0, names='d')
    >>> locals().update(D_blades)

    >>> # Pauli Algebra  `P`
    >>> P, P_blades = Cl(3, names='p')
    >>> locals().update(P_blades)
    >>> sta_split = BladeMap([(d01, p1),
                              (d02, p2),
                              (d03, p3),
                              (d12, p12),
                              (d23, p23),
                              (d13, p13)])

    '''
    def __init__(self, blades_map, map_scalars=True):
        self.blades_map = blades_map

        if map_scalars:
            # make scalars in each algebra map
            s1 = self.b1[0]._newMV(dtype=int)+1
            s2 = self.b2[0]._newMV(dtype=int)+1
            self.blades_map = [(s1, s2)] + self.blades_map

        first_src, first_dest = blades_map[0]
        super().__init__(first_src.layout, first_dest.layout)

    @property
    def b1(self):
        return [k[0] for k in self.blades_map]

    @property
    def b2(self):
        return [k[1] for k in self.blades_map]

    @property
    def layout1(self):
        return self.src_layout

    @property
    def layout2(self):
        return self.dst_layout

    def __call__(self, A):
        '''map an MV `A` according to blade_map'''

        # determine direction of map
        if A.layout == self.src_layout:
            from_b = self.b1
            to_b = self.b2

        elif A.layout == self.dst_layout:
            from_b = self.b2
            to_b = self.b1
        else:
            raise ValueError('A doesnt belong to either Algebra in this Map')

        # create empty MV, and map values
        B = to_b[0]._newMV(dtype=int)
        for from_obj, to_obj in zip(from_b, to_b):
            B += (sum(A.value*from_obj.value)*to_obj)
        return B
