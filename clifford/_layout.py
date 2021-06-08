import functools
from typing import List, Tuple, Optional, Dict, Container
import warnings

import numpy as np
import sparse

# TODO: move some of these functions to this file if they're not useful anywhere
# else
import clifford as cf
from . import (
    get_mult_function,
    val_get_left_gmt_matrix,
    val_get_right_gmt_matrix,
    _numba_val_get_left_gmt_matrix,
    NUMBA_PARALLEL
)
from . import _numba_utils
from .io import read_ga_file
from . import _settings
from ._multivector import MultiVector
from ._layout_helpers import (
    BasisBladeOrder, BasisVectorIds, canonical_reordering_sign_euclidean
)


class _cached_property:
    def __init__(self, getter):
        self.fget = getter
        self.__name__ = getter.__name__
        self.__doc__ = getter.__doc__

    def __get__(self, obj, cls):
        if obj is None:
            return self
        val = self.fget(obj)
        # this entry hides the _cached_property
        setattr(obj, self.__name__, val)
        return val


@_numba_utils.njit(cache=True)
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


@_numba_utils.njit(cache=True)
def gmt_element(bitmap_a, bitmap_b, sig_array):
    """
    Element of the geometric multiplication table given blades a, b.
    The implementation used here is described in :cite:`ga4cs` chapter 19.
    """
    output_sign = canonical_reordering_sign(bitmap_a, bitmap_b, sig_array)
    output_bitmap = bitmap_a^bitmap_b
    return output_bitmap, output_sign


@_numba_utils.njit(cache=True)
def imt_check(grade_v, grade_i, grade_j):
    """
    A check used in imt table generation
    """
    # A_r . B_s = <A_r B_s>_|r-s|
    # if r, s != 0
    return (grade_v == abs(grade_i - grade_j)) and (grade_i != 0) and (grade_j != 0)


@_numba_utils.njit(cache=True)
def omt_check(grade_v, grade_i, grade_j):
    """
    A check used in omt table generation
    """
    # A_r ^ B_s = <A_r B_s>_|r+s|
    return grade_v == (grade_i + grade_j)


@_numba_utils.njit(cache=True)
def lcmt_check(grade_v, grade_i, grade_j):
    """
    A check used in lcmt table generation
    """
    # A_r _| B_s = <A_r B_s>_(s-r) if s-r >= 0
    return grade_v == (grade_j - grade_i)


@_numba_utils.njit(parallel=NUMBA_PARALLEL, nogil=True, cache=True)
def _numba_construct_gmt(
    index_to_bitmap, bitmap_to_index, signature
):
    n = len(index_to_bitmap)
    array_length = int(n * n)
    coords = np.zeros((3, array_length), dtype=np.uint64)
    k_list = coords[0, :]
    l_list = coords[1, :]
    m_list = coords[2, :]

    # use as small a type as possible to minimize type promotion
    mult_table_vals = np.zeros(array_length, dtype=np.int8)

    for i in range(n):
        bitmap_i = index_to_bitmap[i]

        for j in range(n):
            bitmap_j = index_to_bitmap[j]
            bitmap_v, mul = gmt_element(bitmap_i, bitmap_j, signature)
            v = bitmap_to_index[bitmap_v]

            list_ind = i * n + j
            k_list[list_ind] = i
            l_list[list_ind] = v
            m_list[list_ind] = j

            mult_table_vals[list_ind] = mul

    return coords, mult_table_vals


def construct_gmt(
    blade_order: BasisBladeOrder, signature
) -> sparse.COO:
    # wrap the numba one
    coords, mult_table_vals = _numba_construct_gmt(
        blade_order.index_to_bitmap,
        blade_order.bitmap_to_index,
        signature
    )
    dims = len(blade_order.grades)
    return sparse.COO(coords=coords, data=mult_table_vals, shape=(dims, dims, dims))


@_numba_utils.njit(parallel=NUMBA_PARALLEL, nogil=True)
def _numba_construct_graded_mt(
    index_to_grade, coords, gmt_vals, check_func
):
    n_elems = coords.shape[1]

    mask = np.zeros(n_elems, dtype=np.bool_)

    for ind in range(coords.shape[1]):
        k, l, m = coords[:, ind]

        grade_k = index_to_grade[k]
        grade_l = index_to_grade[l]
        grade_m = index_to_grade[m]

        mask[ind] = check_func(grade_l, grade_k, grade_m)

    return coords[:, mask], gmt_vals[mask]


def construct_graded_mt(
    blade_order: BasisBladeOrder, gmt: sparse.COO, check_func
) -> sparse.COO:
    # wrap the numba one
    coords, mult_table_vals = _numba_construct_graded_mt(
        blade_order.grades,
        gmt.coords,
        gmt.data,
        check_func
    )
    dims = len(blade_order.grades)
    return sparse.COO(coords=coords, data=mult_table_vals, shape=(dims, dims, dims))


class Layout(object):
    r""" Layout stores information regarding the geometric algebra itself and the
    internal representation of multivectors.

    Parameters
    ----------

    sig : List[int]
        The signature of the vector space.  This should be
        a list of positive and negative numbers where the sign determines the
        sign of the inner product of the corresponding vector with itself.
        The values are irrelevant except for sign.  This list also determines
        the dimensionality of the vectors.

        Examples::

            sig=[+1, -1, -1, -1] # Hestenes', et al. Space-Time Algebra
            sig=[+1, +1, +1]     # 3-D Euclidean signature

    ids : Optional[BasisVectorIds[Any]]
        A list of ids to associate with each basis vector. These ids are used
        to generate names (if not passed explicitly), and also used when using
        tuple-notation to access elements, such as ``mv[(1, 3)] = 1``.
        Defaults to ``BasisVectorIds.ordered_integers(len(sig))``; that is,
        integers starting at 1.
        This supersedes the old `firstIdx` argument.

        Examples::

            ids=BasisVectorIds.ordered_integers(2, first_index=1)
            ids=BasisVectorIds([10, 20, 30])

        .. versionadded:: 1.3.0

    order : Optional[BasisBladeOrder]
        A specification of the memory order to use when storing the basis blades.
        Defaults to ``BasisBladeOrder.shortlex(len(sig))``.
        This supersedes the old `bladeTupList` argument.

        .. warning::
            Various tools within clifford assume this default, so do not change
            this unless you know what you're doing!

        .. versionadded:: 1.3.0

    bladeTupList : List[Tuple[int, ...]]
        List of tuples corresponding to the blades in the whole
        algebra.  This list determines the order of coefficients in the
        internal representation of multivectors.  The entry for the scalar
        must be an empty tuple, and the entries for grade-1 vectors must be
        singleton tuples.  Remember, the length of the list will be ``2**dims`.

        Example::

            bladeTupList = [(), (0,), (1,), (0, 1)]  # 2-D

        .. deprecated:: 1.3.0

            Use the new `order` and `ids` arguments instead. The above example
            can be spelt with the slightly longer::

                ids = BasisVectorIds([.ordered_integers(2, first_index=0)])
                order = ids.order_from_tuples([(), (0,), (1,), (0, 1)])
                Layout(sig, ids=ids, order=order)

    firstIdx : int
        The index of the first vector.  That is, some systems number
        the base vectors starting with 0, some with 1.

        .. deprecated:: 1.3.0

            Use the new `ids` argument instead, for which the docs show an
            equivalent replacement

    names : List[str]
        List of names of each blade.  When pretty-printing multivectors,
        use these symbols for the blades.  names should be in the same order
        as `order`.  You may use an empty string for scalars.  By
        default, the name for each non-scalar blade is 'e' plus the ids
        of the blade as given in `ids`.

        Example::

            names=['', 's0', 's1', 'i']  # 2-D

    Attributes
    ----------

    dims :
        dimensionality of vectors (``len(self.sig)``)
    sig :
        normalized signature, with all values ``+1`` or ``-1``
    bladeTupList :
        list of blades
    gaDims :
        2**dims
    names :
        pretty-printing symbols for the blades
    """
    # old signature
    def __init__(self, sig, bladeTupList, firstIdx=1, names=None):
        return sig, bladeTupList, firstIdx, names  # lgtm [py/explicit-return-in-init]
    _legacy_init_parser = __init__

    # new signature
    def __init__(self, sig, *, ids=None, order=None, names=None):
        if ids is not None and not isinstance(ids, BasisVectorIds):
            raise TypeError("ids must be a BasisVectorIds")
        if order is not None and not isinstance(order, BasisBladeOrder):
            raise TypeError("order must be a BasisBladeOrder")
        return sig, ids, order, names  # lgtm [py/explicit-return-in-init]
    _new_init_parser = __init__

    @functools.wraps(_new_init_parser)
    def __init__(self, *args, **kw):
        # handle old vs new arguments. Once we drop support for the old we can
        # eliminate the entire `except` clause here.
        try:
            sig, ids, order, names = self._new_init_parser(*args, **kw)
        except TypeError as e_new:
            # try the old arguments
            try:
                sig, bladeTupList, firstIdx, names = self._legacy_init_parser(*args, **kw)
            except TypeError:
                # if both fail, give the error message about the new one
                raise e_new from None

            import inspect
            warnings.warn(
                "The Layout{} constructor is deprecated, use Layout{} "
                "instead.".format(
                    inspect.signature(self._legacy_init_parser),
                    inspect.signature(self._new_init_parser)
                ), DeprecationWarning, stacklevel=2)

            ids = BasisVectorIds.ordered_integers(len(sig), first_index=firstIdx)
            del firstIdx

            # shortcut the lazy property, no need to recompute this
            order = ids.order_from_tuples(bladeTupList)
            self.bladeTupList = bladeTupList
        else:
            # typically there's no need to override this
            if ids is None:
                ids = BasisVectorIds.ordered_integers(len(sig))
            if order is None:
                order = BasisBladeOrder.shortlex(len(sig))

            if len(ids.values) != len(sig):
                raise ValueError(
                    "Length of basis vector ids must match length of signature")

        self.dims = len(sig)
        self.sig = np.array(sig).astype(int)

        self._basis_vector_ids = ids
        self._basis_blade_order = order

        self.gaDims = len(order.grades)

        self._metric = None

        if names is None or isinstance(names, str):
            if isinstance(names, str):
                e = names
            else:
                e = 'e'
            self.names = [
                e + ''.join(map(str, tup)) if tup else ''
                for tup in self.bladeTupList
            ]

        elif len(names) == self.gaDims:
            self.names = names
        else:
            raise ValueError(
                "names list of length %i needs to be of length %i" %
                (len(names), self.gaDims))

    @property
    def gradeList(self):
        return list(self._basis_blade_order.grades)

    @_cached_property
    def gmt(self):
        r""" Multiplication table for the geometric product.

        This is a tensor of rank 3 such that
        :math:`a = b c` can be computed as
        :math:`a_j = \sum_{i,k} b_i \mathit{M}_{ijk} c_k`."""
        return construct_gmt(self._basis_blade_order, self.sig)

    @_cached_property
    def omt(self):
        """ Multiplication table for the inner product, stored in the same way as :attr:`gmt` """
        return construct_graded_mt(self._basis_blade_order, self.gmt, omt_check)

    @_cached_property
    def imt(self):
        """ Multiplication table for the outer product, stored in the same way as :attr:`gmt` """
        return construct_graded_mt(self._basis_blade_order, self.gmt, imt_check)

    @_cached_property
    def lcmt(self):
        """ Multiplication table for the left-contraction, stored in the same way as :attr:`gmt` """
        return construct_graded_mt(self._basis_blade_order, self.gmt, lcmt_check)

    @_cached_property
    def bladeTupList(self):
        return self._basis_vector_ids.order_as_tuples(self._basis_blade_order)

    @property
    def firstIdx(self):
        """ Starting point for vector indices

        .. deprecated:: 1.3.0
            This attribute has been deprecated, to match the deprecation of the
            matching argument in the constructor. Internal code should be using
            ``self._basis_vector_ids.values[x]`` instead of
            ``x + self.firstIdx``. This replacement API is not yet finalized,
            so if you need it please file an issue on github!
        """
        try:
            i = self._basis_vector_ids._first_index
        except AttributeError:
            raise AttributeError("'Layout' objects no longer always have a 'firstIdx' attribute") from None
        else:
            warnings.warn(
                "Layout.firstIdx is deprecated, and will be removed. If you "
                "needed access to this, please contact us!",
                DeprecationWarning, stacklevel=2)
            return i

    @classmethod
    def _from_sig(cls, sig=None, *, firstIdx=1, **kwargs):
        """ Factory method that uses sorted blade tuples.  """
        return cls(
            sig,
            ids=BasisVectorIds.ordered_integers(len(sig), first_index=firstIdx),
            order=None,  # use the default
            **kwargs
        )

    @classmethod
    def _from_Cl(cls, p=0, q=0, r=0, **kwargs):
        """ Factory method from a :math:`{Cl}_{p,q,r}` notation """
        return cls._from_sig([0]*r + [+1]*p + [-1]*q, **kwargs)

    def __hash__(self):
        """ hashes the signature of the layout """
        return hash(tuple(self.sig))

    @_cached_property
    def dual_func(self):
        """ Generates the dual function for the pseudoscalar """
        if 0 in self.sig:
            # We are degenerate, use the right complement
            return self.right_complement_func
        else:
            # Equivalent to but faster than
            #   Iinv = self.pseudoScalar.inv().value
            II_scalar = self.gmt[-1, 0, -1]
            inv_II_scalar = 1 / II_scalar
            if II_scalar in (1, -1):
                Iinv = np.zeros(self.gaDims, dtype=int)
            else:
                Iinv = np.zeros(self.gaDims, dtype=type(inv_II_scalar))
            # set the pseudo-scalar part
            Iinv[-1] = inv_II_scalar

            gmt_func = self.gmt_func
            @_numba_utils.njit
            def dual_func(Xval):
                return gmt_func(Xval, Iinv)
            return dual_func

    @_cached_property
    def _grade_invol(self):
        """
        Generates the grade involution function
        """
        signs = np.power(-1, self._basis_blade_order.grades)
        @_numba_utils.njit
        def grade_inv_func(mv):
            newValue = signs * mv.value
            return self.MultiVector(newValue)
        return grade_inv_func

    @_cached_property
    def vee_func(self):
        """
        Generates the vee product function
        """
        # Often, the dual and undual are used here. However, this unecessarily
        # invokes the metric for a product that is itself non-metric. The
        # complement functions are faster anyway.
        rc_func = self.right_complement_func
        lc_func = self.left_complement_func
        omt_func = self.omt_func
        @_numba_utils.njit
        def vee(aval, bval):
            return lc_func(omt_func(rc_func(aval), rc_func(bval)))
        return vee

    def __repr__(self):
        return "{}({!r}, ids={!r}, order={!r}, names={!r})".format(
            type(self).__name__,
            list(self.sig), self._basis_vector_ids, self._basis_blade_order, self.names
        )

    def _repr_pretty_(self, p, cycle):
        if cycle:
            raise RuntimeError("Should not be cyclic")

        prefix = '{}('.format(type(self).__name__)

        with p.group(len(prefix), prefix, ')'):
            p.text('{},'.format(list(self.sig)))
            p.breakable()
            p.text('ids=')
            p.pretty(self._basis_vector_ids)
            p.text(',')
            p.breakable()
            p.text('order=')
            p.pretty(self._basis_blade_order)
            p.text(',')
            p.breakable()
            p.text('names={}'.format(self.names))

    def __eq__(self, other):
        if other is self:
            return True
        elif isinstance(other, Layout):
            return np.array_equal(self.sig, other.sig)
        else:
            return NotImplemented

    def parse_multivector(self, mv_string: str) -> MultiVector:
        """ Parses a multivector string into a MultiVector object """
        # guarded import in case the parse become heavier weight
        from ._parser import parse_multivector
        return parse_multivector(self, mv_string)

    def gmt_func_generator(self, grades_a=None, grades_b=None, filter_mask=None):
        return get_mult_function(
            self.gmt, self._basis_blade_order.grades,
            grades_a=grades_a, grades_b=grades_b, filter_mask=filter_mask
        )

    def imt_func_generator(self, grades_a=None, grades_b=None, filter_mask=None):
        return get_mult_function(
            self.imt, self._basis_blade_order.grades,
            grades_a=grades_a, grades_b=grades_b, filter_mask=filter_mask
        )

    def omt_func_generator(self, grades_a=None, grades_b=None, filter_mask=None):
        return get_mult_function(
            self.omt, self._basis_blade_order.grades,
            grades_a=grades_a, grades_b=grades_b, filter_mask=filter_mask
        )

    def lcmt_func_generator(self, grades_a=None, grades_b=None, filter_mask=None):
        return get_mult_function(
            self.lcmt, self._basis_blade_order.grades,
            grades_a=grades_a, grades_b=grades_b, filter_mask=filter_mask
        )

    def get_grade_projection_matrix(self, grade):
        """
        Returns the matrix M_g that performs grade projection via left multiplication
        eg. ``M_g@A.value = A(g).value``
        """
        diag_mask = 1.0 * (self._basis_blade_order.grades == grade)
        return np.diag(diag_mask)

    def _gen_complement_func(self, omt):
        """
        Generates the function which computes the complement of a multivector.

        `omt` should be an outer product table.
        """
        dims = self.gaDims
        signlist = np.zeros(dims)

        # Since we're working with basis blades, we can use the table directly.
        # We only care about the pseudo-scalar part of the wedge.
        omt_ps_part = omt[:, -1, :]
        for n in range(dims):
            signlist[n] = (-1)**(omt_ps_part[n, dims-1-n] < 0.001)

        @_numba_utils.njit
        def comp_func(Xval):
            Yval = np.zeros(dims, dtype=Xval.dtype)
            for i, s in enumerate(signlist):
                Yval[i] = Xval[dims-1-i]*s
            return Yval
        return comp_func

    @_cached_property
    def _shirokov_inverse(self):
        """
        Performs the inversion operation as described in Theorem 4, page 16 of the paper :cite:`shirokov2020inverse`
        """
        n = len(self.sig)
        exponent = (n + 1) // 2
        N = 2 ** exponent
        @_numba_utils.njit
        def shirokov_inverse(U):
            Uk = U * 1.0  # cast to float
            for k in range(1, N):
                Ck = (N / k) * Uk.value[0]
                adjU = (Uk - Ck)
                Uk = U * adjU
            if Uk.value[0] == 0:
                raise ValueError('Multivector has no inverse')
            return adjU / Uk.value[0]
        return shirokov_inverse

    @_cached_property
    def _hitzer_inverse(self):
        """
        Performs the inversion operation as described in the paper :cite:`Hitzer_Sangwine_2017`
        """
        tot = len(self.sig)
        @_numba_utils.njit
        def hitzer_inverse(operand):
            if tot == 0:
                numerator = 1 + 0*operand
            elif tot == 1:
                # Equation 4.3
                mv_invol = operand.gradeInvol()
                numerator = mv_invol
            elif tot == 2:
                # Equation 5.5
                mv_conj = operand.conjugate()
                numerator = mv_conj
            elif tot == 3:
                # Equation 6.5  without the rearrangement from 6.4
                mv_conj = operand.conjugate()
                mv_mul_mv_conj = operand * mv_conj
                numerator = (mv_conj * ~mv_mul_mv_conj)
            elif tot == 4:
                # Equation 7.7
                mv_conj = operand.conjugate()
                mv_mul_mv_conj = operand * mv_conj
                numerator = mv_conj * (mv_mul_mv_conj - 2 * mv_mul_mv_conj(3, 4))
            elif tot == 5:
                # Equation 8.22 without the rearrangement from 8.21
                mv_conj = operand.conjugate()
                mv_mul_mv_conj = operand * mv_conj
                combo_op = mv_conj * ~mv_mul_mv_conj
                mv_combo_op = operand * combo_op
                numerator = combo_op * (mv_combo_op - 2 * mv_combo_op(1, 4))
            else:
                raise NotImplementedError(
                    'Closed form inverses for algebras with more than 5 dimensions are not implemented')
            denominator = (operand * numerator).value[0]
            if denominator == 0:
                raise ValueError('Multivector has no inverse')
            return numerator / denominator
        return hitzer_inverse

    @_cached_property
    def gmt_func(self):
        return get_mult_function(self.gmt, self._basis_blade_order.grades)

    @_cached_property
    def imt_func(self):
        return get_mult_function(self.imt, self._basis_blade_order.grades)

    @_cached_property
    def omt_func(self):
        return get_mult_function(self.omt, self._basis_blade_order.grades)

    @_cached_property
    def lcmt_func(self):
        return get_mult_function(self.lcmt, self._basis_blade_order.grades)

    @_cached_property
    def left_complement_func(self):
        return self._gen_complement_func(omt=self.omt)

    @_cached_property
    def right_complement_func(self):
        return self._gen_complement_func(omt=self.omt.T)

    @_cached_property
    def adjoint_func(self):
        '''
        This function returns a fast jitted adjoint function
        '''
        grades = self._basis_blade_order.grades
        signs = np.power(-1, grades*(grades-1)//2)
        @_numba_utils.njit
        def adjoint_func(value):
            return signs * value  # elementwise multiplication
        return adjoint_func

    @_cached_property
    def inv_func(self):
        """
        Get a function that returns left-inverse using a computational linear algebra method
        proposed by Christian Perwass.
         -1         -1
        M    where M  * M  == 1
        """
        mult_table = self.gmt
        k_list, l_list, m_list = mult_table.coords
        mult_table_vals = mult_table.data
        n_dims = mult_table.shape[1]

        identity = np.zeros((n_dims,))
        identity[self._basis_blade_order.bitmap_to_index[0]] = 1

        @_numba_utils.njit
        def leftLaInvJIT(value):
            intermed = _numba_val_get_left_gmt_matrix(value, k_list, l_list, m_list, mult_table_vals, n_dims)
            if abs(np.linalg.det(intermed)) < _settings._eps:
                raise ValueError("multivector has no left-inverse")
            sol = np.linalg.solve(intermed, identity.astype(intermed.dtype))
            return sol

        return leftLaInvJIT

    def get_left_gmt_matrix(self, x):
        """
        This produces the matrix X that performs left multiplication with x
        eg. ``X@b == (x*b).value``
        """
        return val_get_left_gmt_matrix(self.gmt, x.value)

    def get_right_gmt_matrix(self, x):
        """
        This produces the matrix X that performs right multiplication with x
        eg. ``X@b == (b*x).value``
        """
        return val_get_right_gmt_matrix(self.gmt, x.value)

    def load_ga_file(self, filename: str) -> 'cf.MVArray':
        """
        Loads the data from a ga file, checking it matches this layout.
        """
        data_array, metric, basis_names, support = read_ga_file(filename)
        if not np.allclose(np.diagonal(metric), self.sig):
            raise ValueError('The signature of the ga file does not match this layout')
        return cf.MVArray.from_value_array(self, data_array)

    def grade_mask(self, grade: int) -> np.ndarray:
        return grade == self._basis_blade_order.grades

    @property
    def rotor_mask(self) -> np.ndarray:
        return self._basis_blade_order.grades % 2 == 0

    @property
    def metric(self) -> np.ndarray:
        basis_vectors = self.basis_vectors_lst
        if self._metric is None:
            self._metric = np.zeros((len(basis_vectors), len(basis_vectors)))
            for i, v in enumerate(basis_vectors):
                for j, v2 in enumerate(basis_vectors):
                    self._metric[i, j] = (v | v2)[()]
            return self._metric.copy()
        else:
            return self._metric.copy()

    @property
    def scalar(self) -> MultiVector:
        '''
        the scalar of value 1, for this GA (a MultiVector object)

        useful for forcing a MultiVector type
        '''
        s = self.MultiVector(dtype=int)
        s[()] = 1
        return s

    @property
    def pseudoScalar(self) -> MultiVector:
        '''
        The pseudoscalar, :math:`I`.
        '''
        return self.blades_list[-1]

    I = pseudoScalar

    def randomMV(self, n=1, **kwargs) -> MultiVector:
        '''
        Convenience method to create a random multivector.

        see `clifford.randomMV` for details
        '''
        return cf.randomMV(layout=self, n=n, **kwargs)

    def randomV(self, n=1, **kwargs) -> MultiVector:
        '''
        generate n random 1-vector s
        '''
        return cf.randomMV(layout=self, n=n, grades=[1], **kwargs)

    def randomRotor(self, **kwargs) -> MultiVector:
        '''
        generate a random Rotor.

        this is created by muliplying an N unit vectors, where N is
        the dimension of the algebra if its even; else its one less.

        '''
        n = self.dims if self.dims % 2 == 0 else self.dims - 1
        R = functools.reduce(cf.gp, self.randomV(n, normed=True, **kwargs))
        return R

    # Helpers to get hold of basis blades of various specifications.
    # For historic reasons, we have a lot of different ways to spell similar ideas.

    def _basis_blade(self, i, mvClass=MultiVector) -> MultiVector:
        ''' get a basis blade with only the element at the given storage index set '''
        v = np.zeros((self.gaDims,), dtype=int)
        v[i] = 1
        return mvClass(self, v)

    def _basis_vector_indices(self):
        for v_id in self._basis_vector_ids.values:
            v_bitmap = self._basis_vector_ids.id_as_bitmap(v_id)
            v_index = self._basis_blade_order.bitmap_to_index[v_bitmap]
            yield v_index

    @property
    def basis_vectors(self) -> Dict[str, MultiVector]:
        '''dictionary of basis vectors'''
        return dict(zip(self.basis_names, self.basis_vectors_lst))

    @property
    def basis_names(self) -> List[str]:
        """
        Get the names of the basis vectors, in the order they are stored.

        .. versionchanged:: 1.3.0
            Returns a list instead of a numpy array
        """
        return [self.names[i] for i in self._basis_vector_indices()]

    @property
    def basis_vectors_lst(self) -> List[MultiVector]:
        """
        Like ``blades_of_grade(1)``, but ordered based on the ``ids`` parameter
        passed at construction.
        """
        return [self._basis_blade(i) for i in self._basis_vector_indices()]

    def blades_of_grade(self, grade: int) -> List[MultiVector]:
        '''
        return all blades of a given grade,
        '''
        return [
            self._basis_blade(i)
            for i, i_grade in enumerate(self._basis_blade_order.grades)
            if i_grade == grade
        ]

    @property
    def blades_list(self) -> List[MultiVector]:
        '''
        List of blades in this layout matching the `order` argument this layout
        was constructed from.
        '''
        return [self._basis_blade(i) for i in range(self.gaDims)]

    @property
    def blades(self):
        return self.bases()

    def bases(self, mvClass=MultiVector, grades: Optional[Container[int]] = None) -> Dict[str, MultiVector]:
        """Returns a dictionary mapping basis element names to their MultiVector
        instances, optionally for specific grades

        if you are lazy,  you might do this to populate your namespace
        with the variables of a given layout.

        >>> locals().update(layout.blades())  # doctest: +SKIP

        .. versionchanged:: 1.1.0
            This dictionary includes the scalar
        """
        return {
            name: self._basis_blade(i, mvClass)
            for i, (name, grade) in enumerate(zip(self.names, self._basis_blade_order.grades))
            if grades is None or grade in grades
        }

    def _sign_and_index_from_tuple(self, blade: Tuple) -> Tuple[int, int]:
        """
        Takes a tuple blade representation and converts it to a canonical
        tuple blade representation
        """
        s, bitmap = self._basis_vector_ids.tuple_as_sign_and_bitmap(blade)
        index = self._basis_blade_order.bitmap_to_index[bitmap]
        return s, index

    def _index_as_tuple(self, idx: int) -> Tuple:
        """ Convert an index into a blade tuple """
        return self._basis_vector_ids.bitmap_as_tuple(
            self._basis_blade_order.index_to_bitmap[idx]
        )

    # this needs to be last else it replaces the type for our annotations!
    def MultiVector(self, *args, **kwargs) -> MultiVector:
        '''
        Create a multivector in this layout

        convenience func to ``MultiVector(layout)``
        '''
        return MultiVector(self, *args, **kwargs)
