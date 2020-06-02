""" Helpers for efficient bit operations

Bitmaps are used in clifford as representations of basis blades, hence the need
for these operations. These are only for internal use.
"""

from typing import Iterator

import numba
import numba.extending
import numba.types

from . import _numba_utils


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


if _numba_utils.DISABLE_JIT:
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
