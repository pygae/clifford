import functools
import operator
from typing import Tuple

import numpy as np

from clifford.io import write_ga_file, read_ga_file  # noqa: F401
from ._layout import Layout
from ._multivector import MultiVector

dual_array = np.vectorize(MultiVector.dual)
normal_array = np.vectorize(MultiVector.normal)
call_array = np.vectorize(MultiVector.__call__)


def _interrogate_nested_mvs(input_array) -> Tuple[Tuple[int, ...], Layout, np.dtype]:
    """
    Calculates the shape of the nested input_array, and gets the associated layout.
    Stops descending when it encounters a MultiVector.
    """
    if not isinstance(input_array, MultiVector):
        nested_shape, layout, dtype = _interrogate_nested_mvs(input_array[0])
        return (len(input_array), *nested_shape), layout, dtype
    else:
        return (), input_array.layout, input_array.value.dtype


def _index_nested_iterable(input_iterable, index):
    """
    Given a nested iterable, input_iterable, return the element given by the
    1d index iterable
    """
    return functools.reduce(operator.getitem, index, input_iterable)


@functools.lru_cache(None)
def _get_vectorized_value_func(dtype):
    return np.vectorize(lambda x: x.value, otypes=[dtype], signature='()->(n)')


class MVArray(np.ndarray):
    """
    MultiVector Array
    """
    def __new__(cls, input_array):

        input_shape, layout, dtype = _interrogate_nested_mvs(input_array)
        # copying this across elementwise is necessary to prevent numpy recursing into the multivector coefficients
        obj = np.empty(input_shape, dtype=object)
        for index in np.ndindex(input_shape):
            obj[index] = _index_nested_iterable(input_array, index)

        self = obj.view(cls)
        return self

    def __array_finalize__(self, obj):
        if obj is None:
            return

    def _get_first_element(self):
        return self[(0,) * self.ndim]

    @property
    def value(self):
        """
        Return an np array of the values of multivectors
        """
        value_dtype = self._get_first_element().value.dtype
        return _get_vectorized_value_func(value_dtype)(self)

    @staticmethod
    def from_value_array(layout, value_array):
        """
        Constructs an array of mvs from a value array
        """
        v_new_mv = np.vectorize(lambda v: layout.MultiVector(v), otypes=[object], signature='(n)->()')
        return MVArray(v_new_mv(value_array))

    def save(self, filename, compression=True, transpose=False,
             sparse=False, support=False, compression_opts=1):
        """
        Saves the array to a ga file
        """
        first_element = self._get_first_element()
        write_ga_file(filename, self.value, first_element.layout.metric,
                      first_element.layout.basis_names,
                      compression=compression, transpose=transpose,
                      sparse=sparse, support=support, compression_opts=compression_opts)

    def sum(self):
        """
        sum elements of this MVArray
        """
        out = self[0]
        for k in self[1:]:
            out += k
        return out

    def gp(self):
        '''
        geometric product of all elements of this MVArray  (like reduce)
        like ``self[0]*self[1]*....self[n]``
        '''
        out = self[0]
        for k in self[1:]:
            out *= k
        return out

    def op(self):
        '''
        outer product of all elements of this MVArray  (like reduce)
        like ``self[0]^self[1]^....self[n]``
        '''
        out = self[0]
        for k in self[1:]:
            out = out^k
        return out

    def normal(self):
        """
        Normalises all elements
        """
        return normal_array(self)

    def dual(self):
        """
        Takes the dual of all elements
        """
        return dual_array(self)

    def __call__(self, A):
        """
        Performs grade projection on all elements
        """
        return call_array(self, A)


def array(obj):
    '''
    an array method like :func:`numpy.array`, but returns a :class:`.MVArray`.

    .. versionchanged:: 1.4
        Now produces 0d arrays when ``obj`` is a single :class:`MultiVector`

    Parameters
    -------------
    obj : MultiVector, list
        a MV or a list of MV's

    Examples
    ----------
    >>> import clifford as cf
    >>> from clifford import g3
    >>> import numpy as np
    >>> np.array([1, 2, 3])*cf.array(g3.e12)
    MVArray([(1^e12), (2^e12), (3^e12)], dtype=object)
    '''
    return MVArray(obj)
