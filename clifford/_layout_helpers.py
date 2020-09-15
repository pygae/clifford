"""
This module provides helpers that describe some aspect of a layout.

The two public classes are:

* BasisBladeOrder
* BasisVectorIds

"""

from typing import TypeVar, Generic, Sequence, Tuple, List, Optional
import numpy as np
import functools
import operator

from . import _numba_utils
from ._bit_helpers import count_set_bits, set_bit_indices


@_numba_utils.njit
def canonical_reordering_sign_euclidean(bitmap_a, bitmap_b):
    """
    Computes the sign for the product of bitmap_a and bitmap_b
    assuming a euclidean metric
    """
    a = bitmap_a >> 1
    sum_value = 0
    while a != 0:
        sum_value = sum_value + count_set_bits(a & bitmap_b)
        a = a >> 1
    if (sum_value & 1) == 0:
        return 1
    else:
        return -1


def _is_unique(x) -> bool:
    return len(x) == len(set(x))


class BasisBladeOrder:
    """ Represents the storage order in memory of basis blade coefficients.

    Bitmaps represent which basis vectors are present in a basis blade. For
    instance, in an algebra with basis vectors :math:`e_w, e_x, e_y, e_z`, the
    basis blade :math:`e_xz` is represented with ``0b1010``. Note that this
    appears reversed because binary numbers are written with the nth bit first.

    Attributes
    ----------
    index_to_bitmap : numpy.ndarray
        An array mapping storage indices to bitmaps.
    bitmap_to_indices : numpy.ndarray
        A reverse mapping of :attr:`index_to_bitmap`. If bitmaps are missing,
        this array contains ``-1``.
    grades : numpy.ndarray
        An array mapping indices to the grade of the basis vector, that is the
        number of bits in the bitmap.

    See also
    --------
    BasisVectorIds.order_from_tuples
    """
    def __init__(self, bitmaps):
        if not _is_unique(bitmaps):
            raise ValueError("blade bitmaps are not unique")
        self.index_to_bitmap = np.array(bitmaps, dtype=int)
        self.grades = np.zeros(len(self.index_to_bitmap))
        for i, bitmap in enumerate(self.index_to_bitmap):
            self.grades[i] = count_set_bits(bitmap)
        # chosen so that no product of basis blades lies outside this mapping
        largest = np.bitwise_or.reduce(self.index_to_bitmap) + 1
        self.bitmap_to_index = np.full(largest, -1, dtype=int)
        self.bitmap_to_index[self.index_to_bitmap] = np.arange(len(self.index_to_bitmap), dtype=int)

    def __repr__(self) -> str:
        bitmap_strs = ['{:b}'.format(bitmap) for bitmap in self.index_to_bitmap]
        max_bits = max((len(s) for s in bitmap_strs), default=0)
        bitmap_strs = ['0b' + s.rjust(max_bits, '0') for s in bitmap_strs]
        return "{}([{}])".format(type(self).__name__, ', '.join(bitmap_strs))

    def __hash__(self) -> int:
        return hash(self.index_to_bitmap.tobytes())

    def __eq__(self, other):
        if self is other:
            return True
        if not isinstance(other, BasisBladeOrder):
            return NotImplemented
        return np.array_equal(self.index_to_bitmap, other.index_to_bitmap)

    @classmethod
    def shortlex(cls, n_vectors: int) -> 'BasisBladeOrder':
        """
        Get an optimized shortlex ordering.

        This sorts basis blades first by grade, and then lexicographically.
        """
        return _ShortLexBasisBladeOrder(n_vectors)


class _ShortLexBasisBladeOrder(BasisBladeOrder):  # lgtm [py/missing-call-to-init]
    def __init__(self, n_vectors: int):
        # deliberately skip the base class init, we can do a little better
        self._n = n_vectors

        # could attempt to optimize this by avoiding going via python integers
        # and copying the raw logic from
        # python/cpython.git:Modules/itertoolsmodule.c@combinations_next.
        from clifford import _powerset
        self.index_to_bitmap = np.empty(2**n_vectors, dtype=int)
        self.grades = np.empty(2**n_vectors, dtype=int)
        self.bitmap_to_index = np.empty(2**n_vectors, dtype=int)

        for i, t in enumerate(_powerset([1 << i for i in range(n_vectors)])):
            bitmap = functools.reduce(operator.or_, t, 0)
            self.index_to_bitmap[i] = bitmap
            self.grades[i] = len(t)
            self.bitmap_to_index[bitmap] = i
            del t  # enables an optimization inside itertools.combinations

    def __repr__(self):
        return 'BasisBladeOrder.shortlex({})'.format(self._n)

    def __eq__(self, other):
        if self is other:
            return True
        if not isinstance(other, _ShortLexBasisBladeOrder):
            return NotImplemented
        return self._n == other._n

    def __reduce__(self):
        return __class__, (self._n,)


IdT = TypeVar('IdT')


class BasisVectorIds(Generic[IdT]):
    """
    Stores ids for the ordered set of basis vectors, typically integers.

    Provides helpers to convert between bitmaps indicating which vectors are
    present in a blade, and tuples of the original ids.

    For example::

        >>> ids = BasisVectorIds([11, 22, 33])
        >>> ids.bitmap_as_tuple(0b110)
        (22, 33)
        >>> sign, bitmap = ids.tuple_as_sign_and_bitmap((33, 22))
        >>> assert sign, bitmap == (-1, 0b110)
    """
    def __init__(self, blade_ids: Sequence[IdT]):
        if not _is_unique(blade_ids):
            raise ValueError("blade ids are not unique")
        self.values = blade_ids

    def bitmap_as_tuple(self, bitmap: int) -> Tuple[IdT]:
        """ Convert a bitmap representation into a tuple of ids. """
        return tuple(self.values[n] for n in set_bit_indices(bitmap))

    def id_as_bitmap(self, id: IdT) -> int:
        """ Convert the id of a single vector into a bitmap representation. """
        try:
            return (1 << self.values.index(id))
        except ValueError:
            raise ValueError("Unknown basis {}".format(id)) from None

    def tuple_as_sign_and_bitmap(self, blade: Tuple[IdT]) -> Tuple[int, int]:
        """ Convert a blade from a tuple of ids into a bitmap representation. """
        bitmap_out = 0
        s = 1
        for b in blade:
            bitmap_b = self.id_as_bitmap(b)
            if bitmap_b & bitmap_out:
                raise ValueError("blade contains repeated basis vector {}".format(b))
            # as we don't allow repeated indices, the euclidean version is fine
            s *= canonical_reordering_sign_euclidean(bitmap_out, bitmap_b)
            bitmap_out ^= bitmap_b
        return s, bitmap_out

    def order_from_tuples(self, blades: Sequence[Tuple[IdT]]) -> BasisBladeOrder:
        """ Produce an ordering from a set of tuples.

        This is the inverse of :meth:`order_as_tuples`.

        >>> ids = BasisVectorIds(['x', 'y'])
        >>> ids.order_from_tuples([(), ('y',), ('x', 'y'), ('x',)])
        BasisBladeOrder([0b00, 0b10, 0b11, 0b01])
        """
        bitmaps = []
        for blade in blades:
            s, bitmap = self.tuple_as_sign_and_bitmap(blade)
            if s != 1:
                raise NotImplementedError(
                    "The blade {} is not canonical, and sign flips in storage "
                    "are not supported. Did you mean {}?"
                    .format(blade, self.bitmap_as_tuple(bitmap))
                )
            bitmaps.append(bitmap)
        return BasisBladeOrder(bitmaps)

    def order_as_tuples(self, ordering: BasisBladeOrder) -> List[Tuple[IdT]]:
        """ Represent an ordering with these ids.

        This is the inverse of :meth:`order_from_tuples`.

        >>> ids = BasisVectorIds(['x', 'y'])
        >>> ids.order_as_tuples(BasisBladeOrder([0b00, 0b10, 0b11, 0b01]))
        [(), ('y',), ('x', 'y'), ('x',)]
        """
        return [self.bitmap_as_tuple(b) for b in ordering.index_to_bitmap]

    @classmethod
    def ordered_integers(cls, n: int, *, first_index: int = 1) -> 'BasisVectorIds[int]':
        """ Create a set of `n` sequential integers as ids, starting from `first_index`. """
        # special type is an optimization
        return _OrderedIntegerBasisVectorIds(n, first_index=first_index)

    def augmented_with(self, n: int) -> 'BasisVectorIds':
        """ Return a new copy with `n` new ids at the end. """
        value = list(self.values)
        next_id = max(value) + 1
        value += list(range(next_id, next_id + n))
        return BasisVectorIds(value)

    def __repr__(self) -> str:
        return '{}({!r})'.format(type(self).__name__, self.values)

    def __reduce__(self):
        return __class__, (self.values,)


class _OrderedIntegerBasisVectorIds(BasisVectorIds[int]):
    def __init__(self, n, first_index=1):
        self._n = n
        self._first_index = first_index
        super().__init__(range(first_index, first_index + n))

    def augmented_with(self, n):
        return _OrderedIntegerBasisVectorIds(self._n + n, first_index=self._first_index)

    def __repr__(self):
        if self._first_index == 1:
            return 'BasisVectorIds.ordered_integers({})'.format(self._n)
        else:
            return 'BasisVectorIds.ordered_integers({}, first_index={})'.format(self._n, self._first_index)

    def __reduce__(self):
        if self._first_index == 1:
            return __class__, (self._n,)
        else:
            return __class__, (self._n, self._first_index)


def layout_short_name(layout) -> Optional[str]:
    """ helper to get the short name of a layout """
    if hasattr(layout, '__name__') and '__module__' in layout.__dict__:
        return "{l.__module__}.{l.__name__}".format(l=layout)
    return None
