import re
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

    def __get__(self, obj, cls):
        if obj is None:
            return self
        val = self.fget(obj)
        # this entry hides the _cached_property
        setattr(obj, self.__name__, val)
        return val


# The blade finding regex for parsing strings of mvs
_blade_pattern = re.compile(r"""
    ((^|\s)-?\s?\d+(\.\d+)?)\s|
    ((^|\+|-)\s?(\d+((e(\+|-))|\.)?(\d+)?)\^e\d+(\s|$))
""", re.VERBOSE)


@_numba_utils.njit
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


@_numba_utils.njit
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


@_numba_utils.njit
def imt_check(grade_list_idx, grade_list_i, grade_list_j):
    """
    A check used in imt table generation
    """
    return ((grade_list_idx == abs(grade_list_i - grade_list_j)) and (grade_list_i != 0) and (grade_list_j != 0))


@_numba_utils.njit
def omt_check(grade_list_idx, grade_list_i, grade_list_j):
    """
    A check used in omt table generation
    """
    return grade_list_idx == (grade_list_i + grade_list_j)


@_numba_utils.njit
def lcmt_check(grade_list_idx, grade_list_i, grade_list_j):
    """
    A check used in lcmt table generation
    """
    return grade_list_idx == (grade_list_j - grade_list_i)


@_numba_utils.njit(parallel=NUMBA_PARALLEL, nogil=True)
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
    blade_order: BasisBladeOrder, signature
) -> Tuple[sparse.COO, sparse.COO, sparse.COO, sparse.COO]:
    # wrap the numba one
    indices, *arrs = _numba_construct_tables(
        blade_order.grades,
        blade_order.index_to_bitmap, blade_order.bitmap_to_index,
        signature
    )
    dims = len(blade_order.grades)
    return tuple(
        sparse.COO(
            coords=indices, data=arr, shape=(dims, dims, dims),
            prune=True
        )
        for arr in arrs
    )


class Layout(object):
    """ Layout stores information regarding the geometric algebra itself and the
    internal representation of multivectors.

    Parameters
    ----------

    signature : List[int]
        The signature of the vector space.  This should be
        a list of positive and negative numbers where the sign determines the
        sign of the inner product of the corresponding vector with itself.
        The values are irrelevant except for sign.  This list also determines
        the dimensionality of the vectors.  Signatures with zeroes are not
        permitted at this time.

        Examples::

            signature=[+1, -1, -1, -1] # Hestenes', et al. Space-Time Algebra
            signature=[+1, +1, +1]     # 3-D Euclidean signature

    ids : Optional[BasisVectorIds[Any]]
        A list of ids to associate with each basis vector. These ids are used
        to generate names (if not passed explicitly), and also used when using
        tuple-notation to access elements, such as `mv[(1, 3)] = 1`.
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
        singleton tuples.  Remember, the length of the list will be 2**dims.

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
        the base vectors starting with 0, some with 1.  Choose by passing
        the correct number as firstIdx.  0 is the default.

        .. deprecated:: 1.3.0

            Use the new `ids` argument instead, for which the docs show an
            equivalent replacement

    names : List[str]
        List of names of each blade.  When pretty-printing multivectors,
        use these symbols for the blades.  names should be in the same order
        as bladeTupList.  You may use an empty string for scalars.  By
        default, the name for each non-scalar blade is 'e' plus the indices
        of the blade as given in bladeTupList.

        Example::

            names=['', 's0', 's1', 'i']  # 2-D


    Attributes
    ----------

    dims :
        dimensionality of vectors (== len(signature))
    sig :
        normalized signature (i.e. all values are +1 or -1)
    bladeTupList :
        list of blades
    gradeList :
        corresponding list of the grades of each blade
    gaDims :
        2**dims
    names :
        pretty-printing symbols for the blades
    gmt :
        multiplication table for geometric product [1]
    imt :
        multiplication table for inner product [1]
    omt :
        multiplication table for outer product [1]
    lcmt :
        multiplication table for the left-contraction [1]

    [1] The multiplication tables are NumPy arrays of rank 3 with indices like
        the tensor g_ijk discussed above.
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

        self.gradeList = list(self._basis_blade_order.grades)
        self.gaDims = len(self.gradeList)

        self._metric = None

        if names is None or isinstance(names, str):
            if isinstance(names, str):
                e = str(names)
            else:
                e = 'e'
            self.names = []

            for i in range(self.gaDims):
                if self.gradeList[i] >= 1:
                    self.names.append(e + ''.join(
                        map(str, self.bladeTupList[i])))
                else:
                    self.names.append('')

        elif len(names) == self.gaDims:
            self.names = names
        else:
            raise ValueError(
                "names list of length %i needs to be of length %i" %
                (len(names), self.gaDims))

        self._genTables()
        # preload these lazy properties. Not doing this would likely be faster.
        self.adjoint_func
        self.left_complement_func
        self.right_complement_func
        self.dual_func
        self.vee_func
        self.inv_func

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
            Iinv = np.zeros(self.gaDims)
            II_scalar = self.gmt[-1, 0, -1]
            # set the pseudo-scalar part
            Iinv[-1] = 1 / II_scalar

            gmt_func = self.gmt_func
            @_numba_utils.njit
            def dual_func(Xval):
                return gmt_func(Xval, Iinv)
            return dual_func

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
        # Get the names of the canonical blades
        blade_name_index_map = {name: index for index, name in enumerate(self.names)}

        # Clean up the input string a bit
        cleaned_string = re.sub('[()]', '', mv_string)

        # Create a multivector
        mv_out = MultiVector(self)

        # Apply the regex
        for m in _blade_pattern.finditer(cleaned_string):
            # Clean up the search result
            cleaned_match = m.group(0)

            # Split on the '^'
            stuff = cleaned_match.split('^')

            if len(stuff) == 2:
                # Extract the value of the blade and the index of the blade
                blade_val = float("".join(stuff[0].split()))
                blade_index = blade_name_index_map[stuff[1].strip()]
                mv_out[blade_index] = blade_val
            elif len(stuff) == 1:
                # Extract the value of the scalar
                blade_val = float("".join(stuff[0].split()))
                blade_index = 0
                mv_out[blade_index] = blade_val
        return mv_out

    def _genTables(self):
        "Generate the multiplication tables."
        self.gmt, imt_prod_mask, omt_prod_mask, lcmt_prod_mask = construct_tables(
            self._basis_blade_order,
            self.sig
        )
        self.omt = sparse.where(omt_prod_mask, self.gmt, self.gmt.dtype.type(0))
        self.imt = sparse.where(imt_prod_mask, self.gmt, self.gmt.dtype.type(0))
        self.lcmt = sparse.where(lcmt_prod_mask, self.gmt, self.gmt.dtype.type(0))

        # This generates the functions that will perform the various products
        self.gmt_func = get_mult_function(self.gmt, self.gradeList)
        self.imt_func = get_mult_function(self.imt, self.gradeList)
        self.omt_func = get_mult_function(self.omt, self.gradeList)
        self.lcmt_func = get_mult_function(self.lcmt, self.gradeList)

        # these are probably not useful, but someone might want them
        self.imt_prod_mask = imt_prod_mask
        self.omt_prod_mask = omt_prod_mask
        self.lcmt_prod_mask = lcmt_prod_mask

    def gmt_func_generator(self, grades_a=None, grades_b=None, filter_mask=None):
        return get_mult_function(
            self.gmt, self.gradeList,
            grades_a=grades_a, grades_b=grades_b, filter_mask=filter_mask
        )

    def imt_func_generator(self, grades_a=None, grades_b=None, filter_mask=None):
        return get_mult_function(
            self.imt, self.gradeList,
            grades_a=grades_a, grades_b=grades_b, filter_mask=filter_mask
        )

    def omt_func_generator(self, grades_a=None, grades_b=None, filter_mask=None):
        return get_mult_function(
            self.omt, self.gradeList,
            grades_a=grades_a, grades_b=grades_b, filter_mask=filter_mask
        )

    def lcmt_func_generator(self, grades_a=None, grades_b=None, filter_mask=None):
        return get_mult_function(
            self.lcmt, self.gradeList,
            grades_a=grades_a, grades_b=grades_b, filter_mask=filter_mask
        )

    def get_grade_projection_matrix(self, grade):
        """
        Returns the matrix M_g that performs grade projection via left multiplication
        eg. M_g@A.value = A(g).value
        """
        diag_mask = 1.0 * (np.array(self.gradeList) == grade)
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
            Yval = np.zeros(dims)
            for i, s in enumerate(signlist):
                Yval[i] = Xval[dims-1-i]*s
            return Yval
        return comp_func

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
        grades = np.array(self.gradeList)
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
        identity[self.gradeList.index(0)] = 1

        @_numba_utils.njit
        def leftLaInvJIT(value):
            intermed = _numba_val_get_left_gmt_matrix(value, k_list, l_list, m_list, mult_table_vals, n_dims)
            if abs(np.linalg.det(intermed)) < _settings._eps:
                raise ValueError("multivector has no left-inverse")
            sol = np.linalg.solve(intermed, identity)
            return sol

        return leftLaInvJIT

    def get_left_gmt_matrix(self, x):
        """
        This produces the matrix X that performs left multiplication with x
        eg. X@b == (x*b).value
        """
        return val_get_left_gmt_matrix(self.gmt, x.value)

    def get_right_gmt_matrix(self, x):
        """
        This produces the matrix X that performs right multiplication with x
        eg. X@b == (b*x).value
        """
        return val_get_right_gmt_matrix(self.gmt, x.value)

    def MultiVector(self, *args, **kwargs) -> MultiVector:
        '''
        Create a multivector in this layout

        convenience func to MultiVector(layout)
        '''
        return MultiVector(self, *args, **kwargs)

    def load_ga_file(self, filename):
        """
        Takes a ga file
        Checks it is the same signature as this layout
        Loads the data into an MVArray
        """
        data_array, metric, basis_names, support = read_ga_file(filename)
        if not np.allclose(np.diagonal(metric), self.sig):
            raise ValueError('The signature of the ga file does not match this layout')
        return cf.MVArray.from_value_array(self, data_array)

    def grade_mask(self, grade):
        return np.equal(grade, self.gradeList)

    @property
    def rotor_mask(self):
        return sum(
            self.grade_mask(i)
            for i in range(self.dims + 1)
            if not i % 2
        )

    @property
    def metric(self):
        basis_vectors = self.basis_vectors_lst
        if self._metric is None:
            self._metric = np.zeros((len(basis_vectors), len(basis_vectors)))
            for i, v in enumerate(basis_vectors):
                for j, v2 in enumerate(basis_vectors):
                    self._metric[i, j] = (v | v2)[0]
            return self._metric.copy()
        else:
            return self._metric.copy()

    @property
    def scalar(self):
        '''
        the scalar of value 1, for this GA (a MultiVector object)

        useful for forcing a MultiVector type
        '''
        s = self.MultiVector(dtype=int)
        s[()] = 1
        return s

    @property
    def pseudoScalar(self):
        '''
        the psuedoScalar
        '''
        return self.blades_list[-1]

    I = pseudoScalar

    def randomMV(self, n=1, **kwargs):
        '''
        Convenience method to create a random multivector.

        see `clifford.randomMV` for details
        '''
        return cf.randomMV(layout=self, n=n, **kwargs)

    def randomV(self, n=1, **kwargs):
        '''
        generate n random 1-vector s
        '''
        return cf.randomMV(layout=self, n=n, grades=[1], **kwargs)

    def randomRotor(self):
        '''
        generate a random Rotor.

        this is created by muliplying an N unit vectors, where N is
        the dimension of the algebra if its even; else its one less.

        '''
        n = self.dims if self.dims % 2 == 0 else self.dims - 1
        R = functools.reduce(cf.gp, self.randomV(n, normed=True))
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
    def basis_vectors(self):
        '''dictionary of basis vectors'''
        return dict(zip(self.basis_names, self.basis_vectors_lst))

    @property
    def basis_names(self):
        """
        Get the names of the basis vectors, in the order they are stored.

        .. versionchanged:: 1.3.0
            Returns a list instead of a numpy array
        """
        return [self.names[i] for i in self._basis_vector_indices()]

    @property
    def basis_vectors_lst(self):
        """
        Like ``blades_of_grade(1)``, but ordered based on the ``ids`` parameter
        passed at construction.
        """
        return [self._basis_blade(i) for i in self._basis_vector_indices()]

    def blades_of_grade(self, grade: int) -> List[MultiVector]:
        '''
        return all blades of a given grade,

        Parameters
        ------------
        grade: int
            the desired grade

        Returns
        --------
        blades : list of MultiVectors
        '''
        return [
            self._basis_blade(i)
            for i, i_grade in enumerate(self.gradeList)
            if i_grade == grade
        ]

    @property
    def blades_list(self) -> List[MultiVector]:
        '''
        List of blades in this layout matching the order of `self.bladeTupList`
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

        >>> locals().update(layout.blades())

        .. versionchanged:: 1.1.0
            This dictionary includes the scalar
        """
        return {
            name: self._basis_blade(i, mvClass)
            for i, (name, grade) in enumerate(zip(self.names, self.gradeList))
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
