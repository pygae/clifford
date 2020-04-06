""" Helpers for efficient bit operations

Bitmaps are used in clifford as representations of basis blades, hence the need
for these operations. These are only for internal use.
"""

from typing import Iterator

import numba
import numba.extending
import numba.types
import numba.config


def set_bit_indices(x: int) -> Iterator[int]:
    """ Iterate over the indices of bits set to 1 in `x`, in ascending order """
    n = 0
    while x > 0:
        if x & 1:
            yield n
        x = x >> 1
        n = n + 1


@numba.extending.intrinsic
def __builtin_popcnt(tyctx, x):
    """ Emulate clang and GCC's `__builtin_popcnt` """
    if isinstance(x, numba.types.Integer):
        def impl(cgctx, builder, sig, args):
            x, = args
            return builder.ctpop(x)
        sig = x(x)
        return sig, impl


if numba.config.DISABLE_JIT:
    def count_set_bits(bitmap: int) -> int:
        """ Counts the number of bits set to 1 in bitmap """
        count = 0
        for i in set_bit_indices(bitmap):
            count += 1
        return count

else:
    @numba.njit
    def count_set_bits(x: int) -> int:
        """ Counts the number of bits set to 1 in bitmap """
        return __builtin_popcnt(x)


def left_shift(x, shift):
    if shift >= 0:
        return x << shift
    else:
        return x >> -shift


class BitPermuter:
    """
    Produce a callable that sets bit `i` of out to bit `src_bits[i]` of in.

    Attributes
    ----------
    domain : int
        A mask of the bits that this callable accepts
    range : int
        A mask of the bits that this callable produces
    """
    _inverse = None

    def __init__(self, src_to_dst):
        # find bits which need shifting by the same amount
        self._mask_for_shift = {}
        self.domain = 0
        self.range = 0
        for s, d in src_to_dst.items():
            self._mask_for_shift[d - s] = self._mask_for_shift.setdefault(d - s, 0) | 1 << s
            self.domain |= 1 << s
            self.range |= 1 << d

        # this makes the degenerate case work for numpy arrays
        if not src_to_dst:
            self._mask_for_shift[0] = 0

    def __call__(self, bitmap):
        ret = 0
        for shift, mask in self._mask_for_shift.items():
            ret |= left_shift(bitmap & mask, shift)
        return ret

    def inverse(self, bitmap):
        ret = 0
        for shift, mask in self._mask_for_shift.items():
            ret |= left_shift(bitmap, -shift) & mask
        return ret

    def __repr__(self):
        return "BitPermuter({})".format(self)
