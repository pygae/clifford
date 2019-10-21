import numpy as np


from clifford.io import write_ga_file, read_ga_file  # noqa: F401
from ._multivector import MultiVector

dual_array = np.vectorize(MultiVector.dual)
normal_array = np.vectorize(MultiVector.normal)
call_array = np.vectorize(MultiVector.__call__)


class MVArray(np.ndarray):
    '''
    MultiVector Array
    '''

    def __new__(cls, input_array):
        obj = np.empty(len(input_array), dtype=object)
        obj[:] = input_array
        obj = obj.view(cls)
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return

    @property
    def value(self):
        """
        Return an np array of the values of multivectors
        """
        return np.array([mv.value for mv in self])

    @staticmethod
    def from_value_array(layout, value_array):
        """
        Constructs an array of mvs from a value array
        """
        v_new_mv = np.vectorize(lambda v: MultiVector(layout, v), otypes=[MVArray], signature='(n)->()')
        return MVArray(v_new_mv(value_array))

    def save(self, filename, compression=True, transpose=False,
             sparse=False, support=False, compression_opts=1):
        """
        Saves the array to a ga file
        """
        write_ga_file(filename, self.value, self[0].layout.metric, self[0].layout.basis_names,
                      compression=compression, transpose=transpose,
                      sparse=sparse, support=support, compression_opts=compression_opts)

    def sum(self):
        '''
        sum elements of this MVArray
        '''
        out = self[0]
        for k in self[1:]:
            out += k
        return out

    def gp(self):
        '''
        geometric product of all elements of this MVArray  (like reduce)
        like `self[0]*self[1]*....self[n]`
        '''
        out = self[0]
        for k in self[1:]:
            out *= k
        return out

    def op(self):
        '''
        outer product of all elements of this MVArray  (like reduce)
        like `self[0]^self[1]^....self[n]`
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
