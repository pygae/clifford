import itertools
from functools import reduce

import numpy as np

from ._mvarray import MVArray
from ._multivector import MultiVector
from .operator import op
from . import _settings


class Frame(MVArray):
    '''
    A frame of vectors
    '''
    def __new__(cls, input_array):
        if not np.all([k.grades() == {1} for k in input_array]):
            raise TypeError('Frames must be made from vectors')

        obj = MVArray.__new__(cls, input_array)
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return

    @property
    def En(self) -> MultiVector:
        r'''
        Volume element for this frame, :math:`E_n = e_1 \wedge e_2 \wedge \cdots \wedge e_n`
        '''
        return reduce(op, self)

    @property
    def inv(self) -> 'Frame':
        ''' The inverse frame of self '''

        En = self.En
        # see D&L sec 4.3
        vectors = [
            (-1)**(k)*reduce(op, np.hstack([self[:k], self[k+1:]]))*En.inv()
            for k in range(len(self))]

        return Frame(vectors)

    def is_innermorphic_to(self, other: 'Frame', eps: float = None) -> bool:
        '''
        Is this frame *innermorphic* to  other?

        *innermorphic* means both frames share the same inner-product
        between corresponding vectors. This implies that the two frames
        are related by an orthogonal transform.
        '''
        # make iterable `pairs` of all index combos, without repeat
        pairs = list(itertools.combinations(range(len(self)), 2))
        a, b = self, other
        if eps is None:
            eps = _settings._eps

        return np.array([
            float((b[m]|b[n]) - (a[m]|a[n])) < eps
            for m, n in pairs
        ]).all()
