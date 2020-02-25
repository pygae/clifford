"""
.. currentmodule:: clifford

========================================
clifford (:mod:`clifford`)
========================================

The top-level module.
Provides two core classes, :class:`Layout` and :class:`MultiVector`, along with several helper functions to implement the algebras.

Constructing algebras
=====================

Note that typically the :doc:`predefined-algebras` are sufficient, and there is no need to build an algebra from scratch.

.. autosummary::
    :toctree: generated/

    Cl
    conformalize

Whether you construct your algebras from scratch, or use the predefined ones, you'll end up working with the following types:

.. autosummary::
    :toctree: generated/

    MultiVector
    Layout
    ConformalLayout

Global configuration functions
==============================
These functions are used to change the global behavior of ``clifford``.

.. autosummary::
    :toctree: generated/

    eps
    pretty
    ugly
    print_precision

Miscellaneous classes
=======================

.. autosummary::
    :toctree: generated/

    Frame
    BladeMap

Miscellaneous functions
=======================

.. autosummary::
    :toctree: generated/

    grade_obj
    bases
    randomMV

"""

# Standard library imports.
from functools import reduce
import os
import itertools
from typing import List, Tuple, Set, Container, Dict, Optional

# Major library imports.
import numpy as np
from numpy import linalg
import numba
import sparse


from clifford.io import write_ga_file, read_ga_file  # noqa: F401

from ._version import __version__  # noqa: F401

# For backwards-compatibility. New code should import directly from `clifford.operator`
from .operator import gp, op, ip  # noqa: F401

_eps = 1e-12            # float epsilon for float comparisons
_pretty = True          # pretty-print global
_print_precision = 5    # pretty printing precision on floats


try:
    NUMBA_DISABLE_PARALLEL = os.environ['NUMBA_DISABLE_PARALLEL']
except KeyError:
    NUMBA_PARALLEL = True
else:
    NUMBA_PARALLEL = not bool(NUMBA_DISABLE_PARALLEL)


def linear_operator_as_matrix(func, input_blades, output_blades):
    """
    Return a matrix that performs the operation of the provided linear
    operator function func mapping the input blades to the output blades
    """
    ndimin = len(input_blades)
    ndimout = len(output_blades)
    mat = np.zeros((ndimout, ndimin))
    for i, b in enumerate(input_blades):
        mat[:, i] = np.array([func(b)[j] for j in output_blades])
    return mat


def get_adjoint_function(gradeList):
    '''
    This function returns a fast jitted adjoint function
    '''
    grades = np.array(gradeList)
    signs = np.power(-1, grades*(grades-1)//2)
    @numba.njit
    def adjoint_func(value):
        return signs * value  # elementwise multiplication
    return adjoint_func


@numba.njit(parallel=NUMBA_PARALLEL, nogil=True)
def _numba_construct_tables(
    gradeList, linear_map_to_bitmap, bitmap_to_linear_map, signature
):
    array_length = int(len(gradeList) * len(gradeList))
    indices = np.zeros((3, array_length), dtype=np.uint64)
    k_list = indices[0, :]
    l_list = indices[1, :]
    m_list = indices[2, :]

    imt_prod_mask = np.zeros(array_length, dtype=np.bool_)

    omt_prod_mask = np.zeros(array_length, dtype=np.bool_)

    lcmt_prod_mask = np.zeros(array_length, dtype=np.bool_)

    # use as small a type as possible to minimize type promotion
    mult_table_vals = np.zeros(array_length, dtype=np.int8)

    for i, grade_list_i in enumerate(gradeList):
        blade_bitmap_i = linear_map_to_bitmap[i]

        for j, grade_list_j in enumerate(gradeList):
            blade_bitmap_j = linear_map_to_bitmap[j]
            v, mul = gmt_element(blade_bitmap_i, blade_bitmap_j, signature, bitmap_to_linear_map)

            list_ind = i * len(gradeList) + j
            k_list[list_ind] = i
            l_list[list_ind] = v
            m_list[list_ind] = j

            mult_table_vals[list_ind] = mul
            grade_list_idx = gradeList[v]

            # A_r . B_s = <A_r B_s>_|r-s|
            # if r, s != 0
            imt_prod_mask[list_ind] = imt_check(grade_list_idx, grade_list_i, grade_list_j)

            # A_r ^ B_s = <A_r B_s>_|r+s|
            omt_prod_mask[list_ind] = omt_check(grade_list_idx, grade_list_i, grade_list_j)

            # A_r _| B_s = <A_r B_s>_(s-r) if s-r >= 0
            lcmt_prod_mask[list_ind] = lcmt_check(grade_list_idx, grade_list_i, grade_list_j)

    return indices, mult_table_vals, imt_prod_mask, omt_prod_mask, lcmt_prod_mask


def construct_tables(
    gradeList, linear_map_to_bitmap, bitmap_to_linear_map, signature
) -> Tuple[sparse.COO, sparse.COO, sparse.COO, sparse.COO]:
    # wrap the numba one
    indices, *arrs = _numba_construct_tables(
        gradeList, linear_map_to_bitmap, bitmap_to_linear_map, signature
    )
    dims = len(gradeList)
    return tuple(
        sparse.COO(
            coords=indices, data=arr, shape=(dims, dims, dims),
            prune=True
        )
        for arr in arrs
    )


def get_mult_function(mt: sparse.COO, gradeList,
                      grades_a=None, grades_b=None, filter_mask=None):
    '''
    Returns a function that implements the mult_table on two input multivectors
    '''
    if (filter_mask is None) and (grades_a is not None) and (grades_b is not None):
        # If not specified explicitly, we can specify sparseness by grade
        filter_mask = np.zeros(mt.nnz, dtype=bool)
        k_list, _, m_list = mt.coords
        for i in range(len(filter_mask)):
            if gradeList[k_list[i]] in grades_a:
                if gradeList[m_list[i]] in grades_b:
                    filter_mask[i] = 1
        filter_mask = sparse.COO(coords=mt.coords, data=filter_mask, shape=mt.shape)

    if filter_mask is not None:
        # We can pass the sparse filter mask directly
        mt = sparse.where(filter_mask, mt, mt.dtype.type(0))

        return _get_mult_function(mt)

    else:
        return _get_mult_function_runtime_sparse(mt)


def _get_mult_function_result_type(a: numba.types.Type, b: numba.types.Type, mt: np.dtype):
    a_dt = numba.numpy_support.as_dtype(getattr(a, 'dtype', a))
    b_dt = numba.numpy_support.as_dtype(getattr(b, 'dtype', b))
    return np.result_type(a_dt, mt, b_dt)


def _get_mult_function(mt: sparse.COO):
    """
    Get a function similar to `` lambda a, b: np.einsum('i,ijk,k->j', a, mt, b)``

    Returns
    -------
    func : function (array_like (n_dims,), array_like (n_dims,)) -> array_like (n_dims,)
        A function that computes the appropriate multiplication
    """
    # unpack for numba
    dims = mt.shape[1]
    k_list, l_list, m_list = mt.coords
    mult_table_vals = mt.data

    def get_mult_func(ret_dtype, mult_table_vals_t):
        def mult_inner(value, other_value):
            output = np.zeros(dims, dtype=ret_dtype)
            for k, l, m, val in zip(k_list, l_list, m_list, mult_table_vals_t):
                output[l] += value[k] * val * other_value[m]
            return output
        return mult_inner

    def mv_mult(value, other_value):
        ret_dtype = np.result_type(value.dtype, mt.dtype, other_value.dtype)
        mult_table_vals_t = mult_table_vals.astype(ret_dtype)
        return get_mult_func(ret_dtype, mult_table_vals_t)(value, other_value)

    @numba.generated_jit
    def jit_mv_mult(value, other_value):
        # TODO: this is a workaround for a numba issue when jitting is disabled,
        # we probably need to adjust this when fixed upstream.
        if not isinstance(value, numba.types.Type):
            return mv_mult(value, other_value)

        # this casting will be done at jit-time
        ret_dtype = _get_mult_function_result_type(value, other_value, mult_table_vals.dtype)
        mult_table_vals_t = mult_table_vals.astype(ret_dtype)

        return get_mult_func(ret_dtype, mult_table_vals_t)

    return jit_mv_mult


def _get_mult_function_runtime_sparse(mt: sparse.COO):
    """
    A variant of `_get_mult_function` that attempts to exploit runtime zeros

    The returned function avoids performing multiplications if vectors contain
    zeros.

    TODO: determine if this actually helps.
    """
    # unpack for numba
    dims = mt.shape[1]
    k_list, l_list, m_list = mt.coords
    mult_table_vals = mt.data

    def get_mult_func(ret_dtype, mult_table_vals_t):
        def mult_inner(value, other_value):
            output = np.zeros(dims, dtype=ret_dtype)
            for ind, k in enumerate(k_list):
                v_val = value[k]
                if v_val != 0.0:
                    m = m_list[ind]
                    ov_val = other_value[m]
                    if ov_val != 0.0:
                        l = l_list[ind]
                        output[l] += v_val * mult_table_vals_t[ind] * ov_val
            return output
        return mult_inner

    def mv_mult(value, other_value):
        ret_dtype = np.result_type(value.dtype, mt.dtype, other_value.dtype)
        mult_table_vals_t = mult_table_vals.astype(ret_dtype)
        return get_mult_func(ret_dtype, mult_table_vals_t)(value, other_value)

    @numba.generated_jit
    def jit_mv_mult(value, other_value):
        # TODO: this is a workaround for a numba issue when jitting is disabled,
        # we probably need to adjust this when fixed upstream.
        if not isinstance(value, numba.types.Type):
            return mv_mult(value, other_value)

        # this casting will be done at jit-time
        ret_dtype = _get_mult_function_result_type(value, other_value, mult_table_vals.dtype)
        mult_table_vals_t = mult_table_vals.astype(ret_dtype)
        return get_mult_func(ret_dtype, mult_table_vals_t)

    return jit_mv_mult


@numba.njit
def gmt_element(bitmap_a, bitmap_b, sig_array, bitmap_to_linear_mapping):
    """
    Element of the geometric multiplication table given blades a, b.
    The implementation used here is described in chapter 19 of
    Leo Dorst's book, Geometric Algebra For Computer Science
    """
    output_sign = canonical_reordering_sign(bitmap_a, bitmap_b, sig_array)
    output_bitmap = bitmap_a^bitmap_b
    idx = bitmap_to_linear_mapping[output_bitmap]
    return idx, output_sign


@numba.njit
def imt_check(grade_list_idx, grade_list_i, grade_list_j):
    """
    A check used in imt table generation
    """
    return ((grade_list_idx == abs(grade_list_i - grade_list_j)) and (grade_list_i != 0) and (grade_list_j != 0))


@numba.njit
def omt_check(grade_list_idx, grade_list_i, grade_list_j):
    """
    A check used in omt table generation
    """
    return grade_list_idx == (grade_list_i + grade_list_j)


@numba.njit
def lcmt_check(grade_list_idx, grade_list_i, grade_list_j):
    """
    A check used in lcmt table generation
    """
    return grade_list_idx == (grade_list_j - grade_list_i)


@numba.njit
def grade_obj_func(objin_val, gradeList, threshold):
    """ returns the modal grade of a multivector """
    modal_value_count = np.zeros(objin_val.shape)
    n = 0
    for g in gradeList:
        if np.abs(objin_val[n]) > threshold:
            modal_value_count[g] += 1
        n += 1
    return np.argmax(modal_value_count)


def get_leftLaInv(mult_table, gradeList):
    """
    Get a function that returns left-inverse using a computational linear algebra method
    proposed by Christian Perwass.
     -1         -1
    M    where M  * M  == 1
    """

    k_list, l_list, m_list = mult_table.coords
    mult_table_vals = mult_table.data
    n_dims = mult_table.shape[1]

    identity = np.zeros((n_dims,))
    identity[gradeList.index(0)] = 1

    @numba.njit
    def leftLaInvJIT(value):
        intermed = _numba_val_get_left_gmt_matrix(value, k_list, l_list, m_list, mult_table_vals, n_dims)
        if abs(linalg.det(intermed)) < _eps:
            raise ValueError("multivector has no left-inverse")
        sol = linalg.solve(intermed, identity)
        return sol

    return leftLaInvJIT


def general_exp(x, max_order=15):
    """
    This implements the series expansion of e**mv where mv is a multivector
    The parameter order is the maximum order of the taylor series to use
    """

    result = 1.0
    if max_order == 0:
        return result

    # scale by power of 2 so that its norm is < 1
    max_val = int(np.max(np.abs(x.value)))
    scale = 1
    if max_val > 1:
        max_val <<= 1
    while max_val:
        max_val >>= 1
        scale <<= 1

    scaled = x * (1.0 / scale)

    # taylor approximation
    tmp = 1.0 + 0.0*x
    for i in range(1, max_order):
        if np.any(np.abs(tmp.value) > _eps):
            tmp = tmp*scaled * (1.0 / i)
            result += tmp
        else:
            break

    # undo scaling
    while scale > 1:
        result *= result
        scale >>= 1
    return result


def grade_obj(objin, threshold=0.0000001):
    '''
    Returns the modal grade of a multivector
    '''
    return grade_obj_func(objin.value, np.asarray(objin.layout.gradeList), threshold)


def grades_present(objin: 'MultiVector', threshold=0.0000001) -> Set[int]:
    '''
    Returns all the grades of a multivector with coefficient magnitude bigger than threshold
    '''
    nonzero = abs(objin.value) > threshold
    return {
        grade_i
        for grade_i, nonzero_i in zip(objin.layout.gradeList, nonzero)
        if nonzero_i
    }


@numba.njit
def count_set_bits(bitmap):
    """
    Counts the number of bits set to 1 in bitmap
    """
    bmp = bitmap
    count = 0
    n = 1
    while bmp > 0:
        if bmp & 1:
            count += 1
        bmp = bmp >> 1
        n = n + 1
    return count


@numba.njit
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


@numba.njit
def canonical_reordering_sign(bitmap_a, bitmap_b, metric):
    """
    Computes the sign for the product of bitmap_a and bitmap_b
    given the supplied metric
    """
    bitmap = bitmap_a & bitmap_b
    output_sign = canonical_reordering_sign_euclidean(bitmap_a, bitmap_b)
    i = 0
    while bitmap != 0:
        if (bitmap & 1) != 0:
            output_sign *= metric[i]
        i = i + 1
        bitmap = bitmap >> 1
    return output_sign


def compute_reordering_sign_and_canonical_form(blade, metric, firstIdx):
    """
    Takes a tuple blade representation and converts it to a canonical
    tuple blade representation
    """
    bitmap_out = 0
    s = 1
    for b in blade:
        # split into basis blades, which are always canonical
        bitmap_b = compute_bitmap_representation((b,), firstIdx)
        s *= canonical_reordering_sign(bitmap_out, bitmap_b, metric)
        bitmap_out ^= bitmap_b
    return s, compute_blade_representation(bitmap_out, firstIdx)


def compute_bitmap_representation(blade: Tuple[int, ...], firstIdx: int) -> int:
    """
    Takes a tuple blade representation and converts it to the
    bitmap representation
    """
    bitmap = 0
    for b in blade:
        bitmap = bitmap ^ (1 << (b-firstIdx))
    return bitmap


def compute_blade_representation(bitmap: int, firstIdx: int) -> Tuple[int, ...]:
    """
    Takes a bitmap representation and converts it to the tuple
    blade representation
    """
    bmp = bitmap
    blade = []
    n = firstIdx
    while bmp > 0:
        if bmp & 1:
            blade.append(n)
        bmp = bmp >> 1
        n = n + 1
    return tuple(blade)


# todo: work out how to let numba use the COO objects directly
@numba.njit
def _numba_val_get_left_gmt_matrix(x, k_list, l_list, m_list, mult_table_vals, ndims):
    intermed = np.zeros((ndims, ndims))
    test_ind = 0
    for k in k_list:
        j = l_list[test_ind]
        i = m_list[test_ind]
        intermed[j, i] += mult_table_vals[test_ind] * x[k]
        test_ind = test_ind + 1
    return intermed


def val_get_left_gmt_matrix(mt: sparse.COO, x):
    """
    This produces the matrix X that performs left multiplication with x
    eg. X@b == (x*b).value
    """
    dims = mt.shape[1]
    k_list, l_list, m_list = mt.coords
    return _numba_val_get_left_gmt_matrix(
        x, k_list, l_list, m_list, mt.data, dims
    )


def val_get_right_gmt_matrix(mt: sparse.COO, x):
    """
    This produces the matrix X that performs right multiplication with x
    eg. X@b == (b*x).value
    """
    return val_get_left_gmt_matrix(mt.T, x)


# TODO: Move this to the top once we remove circular imports
from ._layout import Layout  # noqa: E402
from ._multivector import MultiVector  # noqa: E402
from ._conformal_layout import ConformalLayout  # noqa: E402
from ._mvarray import MVArray  # noqa: E402


def array(obj):
    '''
    an array method like numpy.array(), but returns a MVArray

    Parameters
    -------------
    obj : MultiVector, list
        a MV or a list of MV's

    Examples
    ----------
    >>>import clifford as cf
    >>>from clifford import g3
    >>>import numpy as np
    >>>np.random.rand(10)*cf.array(g3.e12)
    '''
    if isinstance(obj, MultiVector):
        # they passed a single MV so make a list of it.
        return MVArray([obj])
    else:
        return MVArray(obj)


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
    def En(self):
        '''
        Volume element for this frame

        En = e1^e2^...^en
        '''
        return reduce(op, self)

    @property
    def inv(self):
        '''
        The inverse frame of self

        Returns
        ---------
        inv : `clifford.Frame`
        '''

        En = self.En
        # see D&L sec 4.3
        vectors = [
            (-1)**(k)*reduce(op, np.hstack([self[:k], self[k+1:]]))*En.inv()
            for k in range(len(self))]

        return Frame(vectors)

    def is_innermorphic_to(self, other, eps=None):
        '''
        Is this frame `innermorhpic` to  other?

        *innermorphic* means both frames share the same inner-product
        between corresponding vectors. This implies that the two frames
        are related by an orthogonal transform

        Parameters
        ------------
        other : `clifford.Frame`
            the other frame

        Returns
        ----------
        value : bool

        '''
        # make iterable `pairs` of all index combos, without repeat
        pairs = list(itertools.combinations(range(len(self)), 2))
        a, b = self, other
        if eps is None:
            eps = _eps

        return np.array([
            float((b[m]|b[n]) - (a[m]|a[n])) < eps
            for m, n in pairs
        ]).all()


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


# copied from the itertools docs
def _powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return itertools.chain.from_iterable(
        itertools.combinations(s, r) for r in range(len(s) + 1)
    )


def elements(dims: int, firstIdx=0) -> List[Tuple[int, ...]]:
    """Return a list of tuples representing all 2**dims of blades
    in a dims-dimensional GA.

    Elements are sorted lexicographically.
    """
    return list(_powerset(range(firstIdx, firstIdx + dims)))


def Cl(p=0, q=0, r=0, sig=None, names=None, firstIdx=1, mvClass=MultiVector):
    r"""Returns a :class:`Layout` and basis blade :class:`MultiVector`\ s for the geometric algebra :math:`Cl_{p,q,r}`.

    The notation :math:`Cl_{p,q,r}` means that the algebra is :math:`p+q+r`-dimensional, with the first :math:`p` vectors with positive signature, the next :math:`q` vectors negative, and the final :math:`r` vectors with null signature.

    Returns
    =======
    layout : Layout
        The resulting layout
    blades : Dict[str, MultiVector]
        The blades of the returned layout, equivalent to ``layout.blades``.
    """
    if sig is None:
        layout = Layout._from_Cl(p, q, r, firstIdx=firstIdx, names=names)
    else:
        layout = Layout._from_sig(sig, firstIdx=firstIdx, names=names)
    return layout, layout.bases(mvClass=mvClass)


def bases(layout, mvClass=MultiVector, grades: Optional[Container[int]] = None) -> Dict[str, MultiVector]:
    """Returns a dictionary mapping basis element names to their MultiVector
    instances, optionally for specific grades

    if you are lazy,  you might do this to populate your namespace
    with the variables of a given layout.

    >>> locals().update(layout.blades())

    .. versionchanged:: 1.1.0
        This dictionary includes the scalar
    """

    dict = {}
    for i in range(layout.gaDims):
        grade = layout.gradeList[i]
        if grades is not None and grade not in grades:
            continue
        v = np.zeros((layout.gaDims,), dtype=int)
        v[i] = 1
        dict[layout.names[i]] = mvClass(layout, v)
    return dict


def basis_vectors(layout):
    '''
    dictionary of basis vectors
    '''
    return bases(layout=layout, grades=[1])


def randomMV(
        layout, min=-2.0, max=2.0, grades=None, mvClass=MultiVector,
        uniform=None, n=1, normed=False):
    """n Random MultiVectors with given layout.

    Coefficients are between min and max, and if grades is a list of integers,
    only those grades will be non-zero.


    Examples
    --------
    >>> randomMV(layout, min=-2.0, max=2.0, grades=None, uniform=None, n=2)

    """

    if n > 1:
        # return many multivectors
        return [randomMV(layout=layout, min=min, max=max, grades=grades,
                         mvClass=mvClass, uniform=uniform, n=1,
                         normed=normed) for k in range(n)]

    if uniform is None:
        uniform = np.random.uniform

    if grades is None:
        mv = mvClass(layout, uniform(min, max, (layout.gaDims,)))
    else:
        if isinstance(grades, int):
            grades = [grades]
        newValue = np.zeros((layout.gaDims,))
        for i in range(layout.gaDims):
            if layout.gradeList[i] in grades:
                newValue[i] = uniform(min, max)
        mv = mvClass(layout, newValue)

    if normed:
        mv = mv.normal()

    return mv


def pretty(precision=None):
    """Makes ``repr(MultiVector)`` default to pretty-print.

    `precision` arg can be used to set the printed precision.

    Parameters
    -----------
    precision : int
        number of sig figs to print past decimal

    Examples
    ----------
    >>> pretty(5)

    """

    global _pretty
    _pretty = True

    if precision is not None:
        print_precision(precision)


def ugly():
    """ Makes ``repr(MultiVector)`` default to eval-able representation. """
    global _pretty
    _pretty = False


def eps(newEps=None):
    """ Get/Set the epsilon for float comparisons. """

    global _eps
    if newEps is not None:
        _eps = newEps
    return _eps


def print_precision(newVal):
    """Set the epsilon for float comparisons.

    Parameters
    -----------
    newVal : int
        number of sig figs to print (see builtin `round`)

    Examples
    ----------
    >>> print_precision(5)
    """

    global _print_precision
    _print_precision = newVal


def conformalize(layout, added_sig=[1, -1], *, mvClass=MultiVector, **kwargs):
    '''
    Conformalize a Geometric Algebra

    Given the `Layout` for a GA of signature (p, q), this
    will produce a GA of signature (p+1, q+1), as well as
    return a new list of blades and some `stuff`. `stuff`
    is a dict containing the null basis blades, and some
    up/down functions for projecting in/out of the CGA.



    Parameters
    -------------
    layout: `clifford.Layout`
        layout of the GA to conformalize (the base)
    added_sig: list-like
        list of +1, -1  denoted the added signatures
    **kwargs :
        passed to Cl() used to generate conformal layout

    Returns
    ---------
    layout_c : :class:`ConformalLayout`
        layout of the base GA
    blades_c : dict
        blades for the CGA
    stuff: dict
        dict mapping the following members of :class:`ConformalLayout` by their
        names, for easy unpacking into the global namespace:

        .. autosummary::

            ~ConformalLayout.ep
            ~ConformalLayout.en
            ~ConformalLayout.eo
            ~ConformalLayout.einf
            ~ConformalLayout.E0
            ~ConformalLayout.I_base
            ~ConformalLayout.up
            ~ConformalLayout.down
            ~ConformalLayout.homo


    Examples
    ---------
    >>> from clifford import Cl, conformalize
    >>> G2, blades = Cl(2)
    >>> G2c, bladesc, stuff = conformalize(G2)
    >>> locals().update(bladesc)
    >>> locals().update(stuff)
    '''
    layout_c = ConformalLayout._from_base_layout(layout, added_sig, **kwargs)
    stuff = {
        attr: getattr(layout_c, attr)
        for attr in [
            "ep", "en", "eo", "einf", "E0",
            "up", "down", "homo", "I_base",
        ]
    }
    return layout_c, layout_c.bases(mvClass=mvClass), stuff


# TODO: fix caching to work
# generate pre-defined algebras and cache them

# sigs = [(1, 1, 0), (2, 0, 0), (3, 1, 0), (3, 0, 0), (3, 2, 0), (4, 0, 0)]
# current_module = sys.modules[__name__]
# caching.build_or_read_cache_and_attach_submods(current_module, sigs=sigs)
