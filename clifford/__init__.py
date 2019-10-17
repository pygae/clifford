"""
.. currentmodule:: clifford
========================================
clifford (:mod:`clifford`)
========================================

The Main Module. Provides two classes, Layout and MultiVector, and several helper functions  to implement the algebras.


Classes
===============

.. autosummary::
    :toctree: generated/

    MultiVector
    Layout
    Frame

Functions
================


.. autosummary::
    :toctree: generated/

    Cl
    conformalize
    grade_obj
    bases
    randomMV
    pretty
    ugly
    eps

"""

# Standard library imports.
from functools import reduce
import os
import re
import math
import numbers
import itertools
from typing import List, Tuple

# Major library imports.
import numpy as np
from numpy import linalg, zeros
import numba
import sparse


from clifford.io import write_ga_file, read_ga_file


__version__ = '1.0.5'

# The blade finding regex for parsing strings of mvs
_blade_pattern = re.compile(r"""
    ((^|\s)-?\s?\d+(\.\d+)?)\s|
    ((^|\+|-)\s?(\d+((e(\+|-))|\.)?(\d+)?)\^e\d+(\s|$))
""", re.VERBOSE)
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
    signs = np.power(-1, grades*(grades-1)/2)
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

    mult_table_vals = np.zeros(array_length, dtype=np.float64)

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


def get_mult_function(mt: sparse.COO, gradeList, product_mask=None,
                      grades_a=None, grades_b=None, filter_mask=None):
    '''
    Returns a function that implements the mult_table on two input multivectors
    '''
    if product_mask is not None:
        mt = sparse.where(product_mask, mt, 0)

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
        mt = sparse.where(filter_mask, mt, 0)

        return _get_mult_function(mt)

    else:
        return _get_mult_function_runtime_sparse(mt)

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

    @numba.njit
    def mv_mult(value, other_value):
        output = np.zeros(dims)
        for ind, k in enumerate(k_list):
            m = m_list[ind]
            l = l_list[ind]
            output[l] += value[k] * mult_table_vals[ind] * other_value[m]
        return output

    return mv_mult


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

    @numba.njit
    def mv_mult(value, other_value):
        output = np.zeros(dims)
        for ind, k in enumerate(k_list):
            v_val = value[k]
            if v_val != 0.0:
                m = m_list[ind]
                ov_val = other_value[m]
                if ov_val != 0.0:
                    l = l_list[ind]
                    output[l] += v_val * mult_table_vals[ind] * ov_val
        return output

    return mv_mult


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
        intermed = np.zeros((n_dims, n_dims))
        for test_ind, i in enumerate(k_list):
            j = l_list[test_ind]
            k = m_list[test_ind]
            intermed[i, j] += mult_table_vals[test_ind] * value[k]
        intermed = np.transpose(intermed)
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
    return grade_obj_func(objin.value, objin.layout.gradeList, threshold)


def grades_present(objin, threshold=0.0000001):
    '''
    Returns all the grades of a multivector with coefficient magnitude bigger than threshold
    '''
    grades = []
    for i in range(objin.layout.gaDims):
        if abs(objin.value[i]) > threshold and \
                objin.layout.gradeList[i] not in grades:
            grades.append(objin.layout.gradeList[i])
    return grades


def generate_blade_tup_map(bladeTupList):
    """
    Generates a mapping from blade tuple to linear index into
    multivector
    """
    blade_map = {}
    for ind, blade in enumerate(bladeTupList):
        blade_map[blade] = ind
    return blade_map


def generate_bitmap_to_linear_index_map(bladeTupList, firstIdx):
    """
    Generates a mapping from the bitmap representation to
    the linear index
    """
    bitmap_map = np.zeros(len(bladeTupList), dtype=int)
    for ind, blade in enumerate(bladeTupList):
        bitmap_map[compute_bitmap_representation(blade, firstIdx)] = ind
    return bitmap_map


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
    blade_out = blade[0]
    s = 1
    for b in blade[1:]:
        s = s*canonical_reordering_sign(blade_out, b, metric)
    return s, compute_blade_representation(compute_bitmap_representation(blade, firstIdx), firstIdx)


def compute_bitmap_representation(blade, firstIdx):
    """
    Takes a tuple blade representation and converts it to the
    bitmap representation
    """
    if len(blade) > 0:
        bitmap = 1 << (blade[0]-firstIdx)
        if len(blade) > 1:
            for b in blade[1:]:
                bitmap = bitmap ^ (1 << (b-firstIdx))
        return bitmap
    else:
        return 0


def compute_blade_representation(bitmap, firstIdx):
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

        Examples:
          signature = [+1, -1, -1, -1] # Hestenes', et al. Space-Time Algebra
          signature = [+1, +1, +1]     # 3-D Euclidean signature

    bladeTupList : List[Tuple[int, ...]]
        List of tuples corresponding to the blades in the whole
        algebra.  This list determines the order of coefficients in the
        internal representation of multivectors.  The entry for the scalar
        must be an empty tuple, and the entries for grade-1 vectors must be
        singleton tuples.  Remember, the length of the list will be 2**dims.

        Example:
          bladeTupList = [(), (0,), (1,), (0, 1)]  # 2-D

    firstIdx : int
        The index of the first vector.  That is, some systems number
        the base vectors starting with 0, some with 1.  Choose by passing
        the correct number as firstIdx.  0 is the default.

    names : List[str]
        List of names of each blade.  When pretty-printing multivectors,
        use these symbols for the blades.  names should be in the same order
        as bladeTupList.  You may use an empty string for scalars.  By
        default, the name for each non-scalar blade is 'e' plus the indices
        of the blade as given in bladeTupList.

        Example:
          names = ['', 's0', 's1', 'i']  # 2-D


    Attributes
    ----------

    dims :
        dimensionality of vectors (== len(signature))
    sig :
        normalized signature (i.e. all values are +1 or -1)
    firstIdx :
        starting point for vector indices
    bladeTupList :
        list of blades
    gradeList :
        corresponding list of the grades of each blade
    gaDims :
        2**dims
    einf :
        if conformal returns einf
    eo :
        if conformal returns eo
    names :
        pretty-printing symbols for the blades
    even :
        dictionary of even permutations of blades to the canonical blades
    odd :
        dictionary of odd permutations of blades to the canonical blades
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

    def __init__(self, sig, bladeTupList, firstIdx=0, names=None):
        self.dims = len(sig)
        self.sig = np.array(sig).astype(int)
        self.firstIdx = firstIdx

        self.bladeTupList = list(map(tuple, bladeTupList))
        self._checkList()

        self.gaDims = len(self.bladeTupList)
        self.gradeList = list(map(len, self.bladeTupList))

        self._metric = None

        self.isconformal = False
        self.einf = None
        self.eo = None

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
        self.adjoint_func = get_adjoint_function(self.gradeList)
        self.right_complement_func = self.gen_right_complement_func()
        self.dual_func = self.gen_dual_func()
        self.vee_func = self.gen_vee_func()

    def __hash__(self):
        """ hashs the signature of the layout """
        return hash(tuple(self.sig))

    def gen_dual_func(self):
        """ Generates the dual function for the pseudoscalar """
        if 0 in self.sig:
            # We are degenerate, use the right complement
            return self.right_complement_func
        else:
            Iinv = self.pseudoScalar.inv().value
            gmt_func = self.gmt_func
            @numba.njit
            def dual_func(Xval):
                return gmt_func(Xval, Iinv)
            return dual_func

    def gen_vee_func(self):
        """
        Generates the vee product function
        """
        dual_func = self.dual_func
        omt_func = self.omt_func
        @numba.njit
        def vee(aval, bval):
            return dual_func(omt_func(dual_func(aval), dual_func(bval)))
        return vee

    @property
    def basis_names(self):
        return np.array(list(sorted(self.basis_vectors.keys())), dtype=bytes)

    def dict_to_multivector(self, dict_in):
        """ Takes a dictionary of coefficient values and converts it into a MultiVector object """
        constructed_values = np.zeros(self.gaDims)
        for k in list(dict_in.keys()):
            constructed_values[int(k)] = dict_in[k]
        return self._newMV(constructed_values)

    def __repr__(self):
        s = ("Layout(%r, %r, firstIdx=%r, names=%r)" % (
                list(self.sig),
                self.bladeTupList, self.firstIdx, self.names))
        return s

    def __eq__(self, other):
        if other is self:
            return True
        elif isinstance(other, Layout):
            return np.array_equal(self.sig, other.sig)
        else:
            return NotImplemented

    def __ne__(self, other):
        if other is self:
            return False
        elif isinstance(other, Layout):
            return not np.array_equal(self.sig, other.sig)
        else:
            return NotImplemented

    def parse_multivector(self, mv_string: str) -> 'MultiVector':
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

    def _checkList(self):
        "Ensure validity of arguments."

        # check for uniqueness
        for blade in self.bladeTupList:
            if self.bladeTupList.count(blade) != 1:
                raise ValueError("blades not unique")

        # check for right dimensionality
        if len(self.bladeTupList) != 2**self.dims:
            raise ValueError("incorrect number of blades")

        # check for valid ranges of indices
        valid = list(range(self.firstIdx, self.firstIdx + self.dims))
        try:
            for blade in self.bladeTupList:
                for idx in blade:
                    if (idx not in valid) or (list(blade).count(idx) != 1):
                        raise ValueError()
        except (ValueError, TypeError):
            raise ValueError("invalid bladeTupList; must be a list of tuples")

    def _genTables(self):
        "Generate the multiplication tables."

        self.bladeTupMap = generate_blade_tup_map(self.bladeTupList)
        self.bitmap_to_linear_map = generate_bitmap_to_linear_index_map(self.bladeTupList, self.firstIdx)
        self.linear_map_to_bitmap = np.zeros(len(self.bladeTupMap), dtype=int)
        for bitmap, linear in enumerate(self.bitmap_to_linear_map):
            self.linear_map_to_bitmap[linear] = int(bitmap)

        self.gmt, imt_prod_mask, omt_prod_mask, lcmt_prod_mask = construct_tables(
            np.array(self.gradeList),
            self.linear_map_to_bitmap,
            self.bitmap_to_linear_map,
            self.sig
        )

        # This generates the functions that will perform the various products
        self.gmt_func = get_mult_function(self.gmt, self.gradeList)
        self.imt_func = get_mult_function(self.gmt, self.gradeList, product_mask=imt_prod_mask)
        self.omt_func = get_mult_function(self.gmt, self.gradeList, product_mask=omt_prod_mask)
        self.lcmt_func = get_mult_function(self.gmt, self.gradeList, product_mask=lcmt_prod_mask)
        self.inv_func = get_leftLaInv(self.gmt, self.gradeList)
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
            self.gmt, self.gradeList,
            grades_a=grades_a, grades_b=grades_b, filter_mask=filter_mask, product_mask=self.imt_prod_mask
        )

    def omt_func_generator(self, grades_a=None, grades_b=None, filter_mask=None):
        return get_mult_function(
            self.gmt, self.gradeList,
            grades_a=grades_a, grades_b=grades_b, filter_mask=filter_mask, product_mask=self.omt_prod_mask
        )

    def lcmt_func_generator(self, grades_a=None, grades_b=None, filter_mask=None):
        return get_mult_function(
            self.gmt, self.gradeList,
            grades_a=grades_a, grades_b=grades_b, filter_mask=filter_mask, product_mask=self.lcmt_prod_mask
        )

    def get_grade_projection_matrix(self, grade):
        """
        Returns the matrix M_g that performs grade projection via left multiplication
        eg. M_g@A.value = A(g).value
        """
        diag_mask = 1.0 * (np.array(self.gradeList) == grade)
        return np.diag(diag_mask)

    def gen_right_complement_func(self):
        """
        Generates the right complement of a multivector
        """
        dims = self.gaDims
        bl = self.blades_list
        signlist = np.zeros(self.gaDims)
        for n in range(len(bl)):
            i = bl[n]
            j = bl[dims-1-n]
            signval = (-1)**((i^j).value[-1] < 0.001)
            signlist[n] = signval

        @numba.njit
        def right_comp_func(Xval):
            Yval = np.zeros(dims)
            for i, s in enumerate(signlist):
                Yval[i] = Xval[dims-1-i]*s
            return Yval
        return right_comp_func

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

    def MultiVector(self, *args, **kw):
        '''
        create a multivector in this layout

        convenience func to Multivector(layout)
        '''
        return MultiVector(layout=self, *args, **kw)

    def load_ga_file(self, filename):
        """
        Takes a ga file
        Checks it is the same signature as this layout
        Loads the data into an MVArray
        """
        data_array, metric, basis_names, support = read_ga_file(filename)
        if not np.allclose(np.diagonal(metric), self.sig):
            raise ValueError('The signature of the ga file does not match this layout')
        return MVArray.from_value_array(self, data_array)

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
        if self._metric is None:
            self._metric = np.zeros((len(self.basis_vectors), len(self.basis_vectors)))
            for i, v in enumerate(self.basis_vectors_lst):
                for j, v2 in enumerate(self.basis_vectors_lst):
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
        return self.MultiVector() + 1

    @property
    def pseudoScalar(self):
        '''
        the psuedoScalar
        '''
        return self.blades_list[-1]

    I = pseudoScalar

    def randomMV(self, n=1, **kw):
        '''
        Convenience method to create a random multivector.

        see `clifford.randomMV` for details
        '''
        kw.update(dict(n=n))
        return randomMV(layout=self, **kw)

    def randomV(self, n=1, **kw):
        '''
        generate n random 1-vector s

        '''
        kw.update(dict(n=n, grades=[1]))
        return randomMV(layout=self, **kw)

    def randomRotor(self):
        '''
        generate a random Rotor.

        this is created by muliplying an N unit vectors, where N is
        the dimension of the algebra if its even; else its one less.

        '''
        n = self.dims if self.dims % 2 == 0 else self.dims - 1
        R = reduce(gp, self.randomV(n, normed=True))
        return R

    @property
    def basis_vectors(self):
        return basis_vectors(self)

    @property
    def basis_vectors_lst(self):
        d = self.basis_vectors
        return [d[k] for k in sorted(d.keys())]

    def blades_of_grade(self, grade):
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
        if grade == 0:
            return self.scalar
        return [k for k in self.blades_list[1:] if k.grades() == [grade]]

    @property
    def blades_list(self):
        '''
        Ordered list of blades in this layout (with scalar as [0])
        '''
        blades = self.blades
        names = self.names
        N = self.gaDims
        return [1.0] + [blades[names[k]] for k in range(1, N)]

    @property
    def blades(self):
        return self.bases()

    def bases(self, *args, **kw):
        '''
        Returns a dictionary mapping basis element names to their MultiVector
        instances, optionally for specific grades

        if you are lazy,  you might do this to populate your namespace
        with the variables of a given layout.

        >>> locals().update(layout.bases())



        See Also
        ---------
        bases
        '''
        return bases(layout=self, *args, **kw)


class MultiVector(object):
    """An  element of the algebra

    Parameters
    -------------
    layout: instance of `clifford.Layout`
        the layout of the algebra

    value : sequence of length layout.gaDims
        the coefficients of the base blades

    Notes
    ------
    The following operators are overloaded as follows:

    * * : geometric product
    * ^ : outer product
    * | : inner product
    * ~ : reversion
    * ||: abs value, this is  sqrt(abs(~M*M))

    sequence method

    * M(N) : grade or subspace projection
    * M[N] : blade projection
    """

    def __init__(self, layout, value=None, string=None, *, dtype: np.dtype = np.float64) -> None:
        """Constructor."""

        self.layout = layout
        self.__array_priority__ = 100

        if value is None:
            if string is None:
                self.value = np.zeros((self.layout.gaDims,), dtype=dtype)
            else:
                self.value = layout.parse_multivector(string).value
        else:
            self.value = np.array(value)
            if self.value.shape != (self.layout.gaDims,):
                raise ValueError(
                    "value must be a sequence of length %s" %
                    self.layout.gaDims)

    def __array__(self) -> 'MVArray':
        # we are a scalar, and the only appropriate dtype is an object array
        return MVArray([self])

    def _checkOther(self, other, coerce=True) -> Tuple['MultiVector', bool]:
        """Ensure that the other argument has the same Layout or coerce value if
        necessary/requested.

        _checkOther(other, coerce=True) --> newOther, isMultiVector
        """
        if isinstance(other, numbers.Number):
            if coerce:
                # numeric scalar
                newOther = self._newMV(dtype=np.result_type(other))
                newOther[()] = other
                return newOther, True
            else:
                return other, False

        elif isinstance(other, MultiVector):
            if other.layout != self.layout:
                raise ValueError(
                    "cannot operate on MultiVectors with different Layouts")
            else:
                return other, True
        else:
            return other, False

    def _newMV(self, newValue=None, *, dtype: np.dtype = None) -> 'MultiVector':
        """Returns a new MultiVector (or derived class instance).
        """
        if newValue is None and dtype is None:
            raise TypeError("Must specify either a type or value")

        return self.__class__(self.layout, newValue, dtype=dtype)

    # numeric special methods
    # binary

    def exp(self) -> 'MultiVector':
        return general_exp(self)

    def vee(self, other) -> 'MultiVector':
        """
        The vee product aka. the meet... To be optimised still
        """
        return self.layout.MultiVector(value=self.layout.vee_func(self.value, other.value))

    def __and__(self, other) -> 'MultiVector':
        """
        The vee product aka. the meet... To be optimised still
        """
        return self.vee(other)

    def __mul__(self, other) -> 'MultiVector':
        """Geometric product

        M * N --> MN
        """

        other, mv = self._checkOther(other, coerce=False)

        if mv:
            newValue = self.layout.gmt_func(self.value, other.value)
        else:
            if isinstance(other, np.ndarray):
                obj = self.__array__()
                return obj*other

            newValue = other * self.value

        return self._newMV(newValue)

    def __rmul__(self, other) -> 'MultiVector':
        """Right-hand geometric product

        N * M --> NM
        """

        other, mv = self._checkOther(other, coerce=False)

        if mv:
            newValue = self.layout.gmt_func(other.value, self.value)
        else:
            if isinstance(other, np.ndarray):
                obj = self.__array__()
                return other*obj
            newValue = other*self.value

        return self._newMV(newValue)

    def __xor__(self, other) -> 'MultiVector':
        """Outer product

        M ^ N
        """

        other, mv = self._checkOther(other, coerce=False)

        if mv:
            newValue = self.layout.omt_func(self.value, other.value)
        else:
            if isinstance(other, np.ndarray):
                obj = self.__array__()
                return obj^other
            newValue = other*self.value

        return self._newMV(newValue)

    def __rxor__(self, other) -> 'MultiVector':
        """Right-hand outer product

        N ^ M
        """

        other, mv = self._checkOther(other, coerce=False)

        if mv:
            newValue = self.layout.omt_func(other.value, self.value)
        else:
            if isinstance(other, np.ndarray):
                obj = self.__array__()
                return other^obj
            newValue = other * self.value

        return self._newMV(newValue)

    def __or__(self, other) -> 'MultiVector':
        """Inner product

        M | N
        """

        other, mv = self._checkOther(other)

        if mv:
            newValue = self.layout.imt_func(self.value, other.value)
        else:
            if isinstance(other, np.ndarray):
                obj = self.__array__()
                return obj|other
            # l * M = M * l = 0 for scalar l
            return self._newMV(dtype=np.result_type(self.value.dtype, other))

        return self._newMV(newValue)

    __ror__ = __or__

    def __add__(self, other) -> 'MultiVector':
        """Addition

        M + N
        """

        other, mv = self._checkOther(other)
        if not mv:
            if isinstance(other, np.ndarray):
                obj = self.__array__()
                return obj + other
        newValue = self.value + other.value

        return self._newMV(newValue)

    __radd__ = __add__

    def __sub__(self, other) -> 'MultiVector':
        """Subtraction

        M - N
        """

        other, mv = self._checkOther(other)
        if not mv:
            if isinstance(other, np.ndarray):
                obj = self.__array__()
                return obj - other
        newValue = self.value - other.value

        return self._newMV(newValue)

    def __rsub__(self, other) -> 'MultiVector':
        """Right-hand subtraction

        N - M
        """

        other, mv = self._checkOther(other)
        if not mv:
            if isinstance(other, np.ndarray):
                obj = self.__array__()
                return other - obj
        newValue = other.value - self.value

        return self._newMV(newValue)

    def right_complement(self) -> 'MultiVector':
        return self.layout.MultiVector(value=self.layout.right_complement_func(self.value))

    def __truediv__(self, other) -> 'MultiVector':
        """Division
                       -1
        M / N --> M * N
        """

        other, mv = self._checkOther(other, coerce=False)

        if mv:
            return self * other.inv()
        else:
            if isinstance(other, np.ndarray):
                obj = self.__array__()
                return obj/other
            newValue = self.value / other
            return self._newMV(newValue)

    def __rtruediv__(self, other) -> 'MultiVector':
        """Right-hand division
                       -1
        N / M --> N * M
        """

        other, mv = self._checkOther(other)
        if isinstance(other, np.ndarray):
            obj = self.__array__()
            return other / obj

        return other * self.inv()

    def __pow__(self, other) -> 'MultiVector':
        """Exponentiation of a multivector by an integer
                    n
        M ** n --> M
        """

        if not isinstance(other, (int, float)):
            raise ValueError("exponent must be a Python int or float")

        if abs(round(other) - other) > _eps:
            raise ValueError("exponent must have no fractional part")

        other = int(round(other))

        if other == 0:
            return 1

        newMV = self._newMV(np.array(self.value))  # copy

        for i in range(1, other):
            newMV = newMV * self

        return newMV

    def __rpow__(self, other) -> 'MultiVector':
        """Exponentiation of a real by a multivector
                  M
        r**M --> r
        """

        # Let math.log() check that other is a Python number, not something
        # else.

        # pow(x, y) == exp(y * log(x))
        newMV = general_exp(math.log(other) * self)

        return newMV

    def __lshift__(self, other) -> 'MultiVector':
        """
        The << operator is the left contraction
        """
        return self.lc(other)

    # unary

    def __neg__(self) -> 'MultiVector':
        """Negation

        -M
        """

        newValue = -self.value

        return self._newMV(newValue)

    def as_array(self) -> np.ndarray:
        return self.value

    def __pos__(self) -> 'MultiVector':
        """Positive (just a copy)

        +M
        """

        newValue = self.value + 0  # copy

        return self._newMV(newValue)

    def mag2(self) -> numbers.Number:
        """Magnitude (modulus) squared
           2
        |M|

        Note in mixed signature spaces this may be negative
        """
        mv_val = self.layout.gmt_func(self.layout.adjoint_func(self.value), self.value)
        return mv_val[0]

    def __abs__(self) -> numbers.Number:
        """Magnitude (modulus)

        abs(M) --> |M|

        This is sqrt(abs(~M*M)).

        The abs inside the sqrt is need for spaces of mixed signature
        """

        return np.sqrt(abs(self.mag2()))

    def adjoint(self) -> 'MultiVector':
        """Adjoint / reversion
               _
        ~M --> M (any one of several conflicting notations)
        ~(N * M) --> ~M * ~N
        """
        # The multivector created by reversing all multiplications
        return self._newMV(self.layout.adjoint_func(self.value))

    __invert__ = adjoint

    # builtin
    def __int__(self) -> int:
        """Coerce to an integer iff scalar.
        """

        return int(self.__float__())

    def __float__(self) -> float:
        """"Coerce to a float iff scalar.
        """

        if self.isScalar():
            return float(self[()])
        else:
            raise ValueError("non-scalar coefficients are non-zero")

    # sequence special methods
    def __len__(self) -> int:
        """Returns length of value array.
        """

        return self.layout.gaDims

    def __getitem__(self, key) -> numbers.Number:
        """If key is a blade tuple (e.g. (0, 1) or (1, 3)), or a blade,
        (e.g. e12),  then return the (real) value of that blade's coefficient.
        Otherwise, treat key as an index into the list of coefficients.

        value = M[blade]
        value = M[index]
        """
        if isinstance(key, MultiVector):
            return self.value[int(np.where(key.value)[0][0])]
        elif key in self.layout.bladeTupMap.keys():
            return self.value[self.layout.bladeTupMap[key]]
        elif isinstance(key, tuple):
            sign, blade = compute_reordering_sign_and_canonical_form(key, np.array(self.layout.sig),
                                                                     self.layout.firstIdx)
            return sign*self.value[self.layout.bladeTupMap[blade]]
        return self.value[key]

    def __setitem__(self, key, value: numbers.Number) -> None:
        """If key is a blade tuple (e.g. (0, 1) or (1, 3)), then set
        the (real) value of that blade's coeficient.
        Otherwise treat key as an index into the list of coefficients.

        M[blade] = value
        M[index] = value
        """
        if key in self.layout.bladeTupMap.keys():
            self.value[self.layout.bladeTupMap[key]] = value
        elif isinstance(key, tuple):
            sign, blade = compute_reordering_sign_and_canonical_form(key, np.array(self.layout.sig),
                                                                     self.layout.firstIdx)
            self.value[self.layout.bladeTupMap[blade]] = sign*value
        else:
            self.value[key] = value

    def __delitem__(self, key) -> None:
        """Set the selected coefficient to 0.

        del M[blade]
        del M[index]
        """

        if key in self.layout.bladeTupMap.keys():
            self.value[self.layout.bladeTupMap[key]] = 0
        elif isinstance(key, tuple):
            sign, blade = compute_reordering_sign_and_canonical_form(key, np.array(self.layout.sig),
                                                                     self.layout.firstIdx)
            self.value[self.layout.bladeTupMap[blade]] = 0
        else:
            self.value[key] = 0

    # grade projection
    def __call__(self, other, *others) -> 'MultiVector':
        """Return a new multi-vector projected onto a grade OR a MV


        M(grade[s]) --> <M>
                        grade
        OR

        M(other) --> other.project(M)

        Examples
        --------
        >>>M(0)
        >>>M(0, 2)
        """
        if isinstance(other, MultiVector):
            return other.project(self)
        else:
            # we are making a grade projection
            grade = other

        if len(others) != 0:
            return sum([self.__call__(k) for k in (other,)+others])

        if grade not in self.layout.gradeList:
            raise ValueError("algebra does not have grade %s" % grade)

        if not np.issubdtype(type(grade), np.integer):
            raise ValueError("grade must be an integer")

        mask = np.equal(grade, self.layout.gradeList)

        newValue = np.multiply(mask, self.value)

        return self._newMV(newValue)

    # fundamental special methods
    def __str__(self) -> str:
        """Return pretty-printed representation.
        """

        s = ''
        p = _print_precision

        for grade, name, coeff in zip(self.layout.gradeList, self.layout.names, self.value):
            # if we have nothing yet, don't use + and - as operators but
            # use - as an unary prefix if necessary
            if s:
                seps = (' + ', ' - ')
            else:
                seps = ('', '-')

            if abs(coeff) >= _eps:
                if coeff > 0:
                    sep = seps[0]
                    abs_coeff = round(coeff, p)
                else:
                    sep = seps[1]
                    abs_coeff = -round(coeff, p)

                if grade == 0:
                    # scalar
                    s = '%s%s%s' % (s, sep, abs_coeff)
                else:
                    # not a scalar
                    s = '%s%s(%s^%s)' % (s, sep, abs_coeff, name)
        if s:
            # non-zero
            return s
        else:
            # return scalar 0
            return '0'

    def __repr__(self) -> str:
        """Return eval-able representation if global _pretty is false.
        Otherwise, return str(self).
        """

        if _pretty:
            return self.__str__()

        s = "MultiVector(%s, value=%s)" % (
            repr(self.layout), list(self.value))
        return s

    def __bool__(self) -> bool:
        """Instance is nonzero iff at least one of the coefficients
        is nonzero.
        """

        nonzeroes = np.absolute(self.value) > _eps

        if nonzeroes.any():
            return True
        else:
            return False

    def __eq__(self, other) -> bool:
        other, mv = self._checkOther(other)
        if not mv:
            return NotImplemented

        if (np.absolute(self.value - other.value) < _eps).all():
            # equal within epsilon
            return True
        else:
            return False

    def __ne__(self, other) -> bool:
        ret = self.__eq__(other)
        if ret is NotImplemented:
            return ret
        return not ret

    def clean(self, eps=None) -> 'MultiVector':
        """Sets coefficients whose absolute value is < eps to exactly 0.

        eps defaults to the current value of the global _eps.

        clean(eps=None)
        """

        if eps is None:
            eps = _eps

        mask = np.absolute(self.value) > eps

        # note element-wise multiplication
        self.value = mask * self.value

        return self

    def round(self, eps=None) -> 'MultiVector':
        """Rounds all coefficients according to Python's rounding rules.

        eps defaults to the current value of the global _eps.

        round(eps=None)
        """

        if eps is None:
            eps = _eps

        self.value = np.around(self.value, eps)

        return self

    # Geometric Algebraic functions
    def lc(self, other) -> 'MultiVector':
        """Returns the left-contraction of two multivectors.

        M _| N
        """

        other, mv = self._checkOther(other, coerce=True)

        newValue = self.layout.lcmt_func(self.value, other.value)

        return self._newMV(newValue)

    @property
    def pseudoScalar(self) -> 'MultiVector':
        "Returns a MultiVector that is the pseudoscalar of this space."
        return self.layout.pseudoScalar

    I = pseudoScalar

    def invPS(self) -> 'MultiVector':
        "Returns the inverse of the pseudoscalar of the algebra."

        ps = self.pseudoScalar

        return ps.inv()

    def isScalar(self) -> bool:
        """Returns true iff self is a scalar.
        """

        indices = list(range(self.layout.gaDims))
        indices.remove(self.layout.gradeList.index(0))

        for i in indices:
            if abs(self.value[i]) < _eps:
                continue
            else:
                return False

        return True

    def isBlade(self) -> bool:
        """Returns true if multivector is a blade.
        From Leo Dorsts GA for computer science section 21.5
        """

        grade = None

        for i in range(self.layout.gaDims):
            if abs(self.value[i]) > _eps:
                if grade is None:
                    grade = self.layout.gradeList[i]
                elif self.layout.gradeList[i] != grade:
                    return False

        Vhat = self.gradeInvol()
        Vrev = ~self
        Vinv = Vrev/(self*Vrev)[0]

        gpres = grades_present(Vhat*Vinv, 0.000001)
        if len(gpres) == 1:
            if gpres[0] == 0:
                if np.sum(np.abs((Vhat*Vinv).value - (Vinv*Vhat).value)) < 0.0001:
                    for e in basis_vectors(self.layout).values():
                        gpres = grades_present(Vhat*e*Vrev, 0.000001)
                        if not (len(gpres) == 1 and gpres[0] == 1):
                            return False
                    return True
        return False

    def isVersor(self) -> bool:
        """Returns true if multivector is a versor.
        From Leo Dorsts GA for computer science section 21.5
        isBlade() --> Boolean
        """

        Vhat = self.gradeInvol()
        Vrev = ~self
        Vinv = Vrev/(self*Vrev)[0]

        gpres = grades_present(Vhat*Vinv, 0.000001)
        if len(gpres) == 1:
            if gpres[0] == 0:
                if np.sum(np.abs((Vhat*Vinv).value - (Vinv*Vhat).value)) < 0.0001:
                    for e in basis_vectors(self.layout).values():
                        gpres = grades_present(Vhat*e*Vrev, 0.000001)
                        if not (len(gpres) == 1 and gpres[0] == 1):
                            return False
                    gpres = grades_present(self, 0.000001)
                    if len(gpres) == 1:
                        return False
                    else:
                        return True
        return False

    def grades(self) -> List[int]:
        """Return the grades contained in the multivector.
        """

        return grades_present(self, _eps)

    @property
    def blades_list(self) -> List['MultiVector']:
        '''
        ordered list of blades present in this MV
        '''
        blades_list = self.layout.blades_list
        value = self.value

        b = [value[0]] + [value[k]*blades_list[k] for k in range(1, len(self))]
        return [k for k in b if k != 0]

    def normal(self) -> 'MultiVector':
        """Return the (mostly) normalized multivector.

        The _mostly_ comes from the fact that some multivectors have a
        negative squared-magnitude.  So, without introducing formally
        imaginary numbers, we can only fix the normalized multivector's
        magnitude to +-1.

        M / |M|  up to a sign
        """

        return self / np.sqrt(abs(self.mag2()))

    def leftLaInv(self) -> 'MultiVector':
        """Return left-inverse using a computational linear algebra method
        proposed by Christian Perwass.
         -1         -1
        M    where M  * M  == 1
        """
        return self._newMV(self.layout.inv_func(self.value))

    def normalInv(self) -> 'MultiVector':
        """Returns the inverse of itself if M*~M == |M|**2.
         -1
        M   = ~M / (M * ~M)
        """

        Madjoint = ~self
        MadjointM = (Madjoint * self)

        if MadjointM.isScalar() and abs(MadjointM[()]) > _eps:
            # inverse exists
            return Madjoint / MadjointM[()]
        else:
            raise ValueError("no inverse exists for this multivector")

    def inv(self) -> 'MultiVector':
        if (self*~self).isScalar():
            it = self.normalInv()
        else:
            it = self.leftLaInv()
        return it

    leftInv = leftLaInv
    rightInv = leftLaInv

    def dual(self, I=None) -> 'MultiVector':
        """Returns the dual of the multivector against the given subspace I.
        I defaults to the pseudoscalar.

        ~        -1
        M = M * I
        """
        if I is None:
            return self.layout.MultiVector(value=self.layout.dual_func(self.value))
        else:
            Iinv = I.inv()

        return self * Iinv

    def commutator(self, other) -> 'MultiVector':
        """Returns the commutator product of two multivectors.

        [M, N] = M X N = (M*N - N*M)/2
        """

        return ((self * other) - (other * self)) / 2

    x = commutator

    def anticommutator(self, other) -> 'MultiVector':
        """Returns the anti-commutator product of two multivectors.

        (M*N + N*M)/2
        """

        return ((self * other) + (other * self)) / 2

    def gradeInvol(self) -> 'MultiVector':
        """Returns the grade involution of the multivector.
         *                    i
        M  = Sum[i, dims, (-1)  <M>  ]
                                   i
        """

        signs = np.power(-1, self.layout.gradeList)

        newValue = signs * self.value

        return self._newMV(newValue)

    @property
    def even(self) -> 'MultiVector':
        '''
        Even part of this mulivector

        defined as
        M + M.gradInvol()
        '''
        return .5*(self + self.gradeInvol())

    @property
    def odd(self) -> 'MultiVector':
        '''
        Odd part of this mulivector

        defined as
        M +- M.gradInvol()
        '''
        return .5*(self - self.gradeInvol())

    def conjugate(self) -> 'MultiVector':
        """Returns the Clifford conjugate (reversion and grade involution).
         *
        M  --> (~M).gradeInvol()
        """

        return (~self).gradeInvol()

    # Subspace operations
    def project(self, other) -> 'MultiVector':
        """Projects the multivector onto the subspace represented by this blade.
                            -1
        P (M) = (M _| A) * A
         A
        """

        other, mv = self._checkOther(other, coerce=True)

        if not self.isBlade():
            raise ValueError("self is not a blade")

        return other.lc(self) * self.inv()

    def factorise(self) -> Tuple[List['MultiVector'], numbers.Number]:
        """
        Factorises a blade into basis vectors and an overall scale
        Uses Leo Dorsts algorithm from 21.6 of GA for Computer Science
        """
        if not self.isBlade():
            raise ValueError("self is not a blade")
        scale = abs(self)
        max_index = np.argmax(np.abs(self.value))
        B_max_factors = self.layout.bladeTupList[max_index]

        factors = []

        B_c = self/scale
        for ind in B_max_factors[1:]:
            ei = self.layout.blades_list[ind]
            fi = (ei.lc(B_c)*(~B_c*(1/(B_c*~B_c)[0]))).normal()
            factors.append(fi)
            B_c = B_c * ~fi * (1 / (fi * ~fi)[0])
        factors.append(B_c.normal())
        factors.reverse()
        return factors, scale

    def basis(self) -> List['MultiVector']:
        """Finds a vector basis of this subspace.
        """
        if not self.isBlade():
            raise ValueError("self is not a blade")

        gr = self.grades()

        selfInv = self.inv()

        selfInv.clean()

        wholeBasis = []  # vector basis of the whole space

        for i in range(self.layout.gaDims):
            if self.layout.gradeList[i] == 1:
                v = np.zeros((self.layout.gaDims,), dtype=float)
                v[i] = 1.
                wholeBasis.append(self._newMV(v))

        thisBasis = []  # vector basis of this subspace

        J, mv = self._checkOther(1.)  # outer product of all of the vectors up
        # to the point of iteration

        for ei in wholeBasis:
            Pei = ei.lc(self) * selfInv

            J.clean()

            J2 = J ^ Pei

            if J2 != 0:
                J = J2
                thisBasis.append(Pei)
                if len(thisBasis) == gr[0]:  # we have a complete set
                    break

        return thisBasis

    def join(self, other) -> 'MultiVector':
        """Returns the join of two blades.
              .
        J = A ^ B
        """

        other, mv = self._checkOther(other)

        grSelf = self.grades()
        grOther = other.grades()

        if len(grSelf) == len(grOther) == 1:
            # both blades

            # try the outer product first
            J = self ^ other

            if J != 0:
                return J.normal()

            # try something else
            M = (other * self.invPS()).lc(self)

            if M != 0:
                C = M.normal()
                J = (self * C.rightInv()) ^ other
                return J.normal()

            if grSelf[0] >= grOther[0]:
                A = self
                B = other
            else:
                A = other
                B = self

            if (A * B) == (A | B):
                # B is a subspace of A or the same if grades are equal
                return A.normal()

            # ugly, but general way
            # watch out for residues

            # A is still the larger-dimensional subspace

            Bbasis = B.basis()

            # add the basis vectors of B one by one to the larger
            # subspace except for the ones that make the outer
            # product vanish

            J = A

            for ei in Bbasis:
                J.clean()
                J2 = J ^ ei

                if J2 != 0:
                    J = J2

            # for consistency's sake, we'll normalize the join
            J = J.normal()

            return J

        else:
            raise ValueError("not blades")

    def meet(self, other, subspace=None) -> 'MultiVector':
        r"""Returns the meet of two blades.

        Computation is done with respect to a subspace that defaults to
        the join if none is given.
                     -1
        M \/i N = (Mi  ) * N
        """

        other, mv = self._checkOther(other)

        r = self.grades()
        s = other.grades()

        if len(r) > 1 or len(s) > 1:
            raise ValueError("not blades")

        if subspace is None:
            subspace = self.join(other)

        return (self * subspace.inv()) | other


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
        if not np.all([k.grades() == [1] for k in input_array]):
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
    """Returns a Layout and basis blades for the geometric algebra Cl_p,q.

    The notation Cl_p,q means that the algebra is p+q dimensional, with
    the first p vectors with positive signature and the final q vectors
    negative.

    Cl(p, q=0, names=None, firstIdx=0) --> Layout, {'name': basisElement, ...}
    """
    if sig is None:
        sig = [0]*r + [+1]*p + [-1]*q
    bladeTupList = elements(len(sig), firstIdx)

    layout = Layout(sig, bladeTupList, firstIdx=firstIdx, names=names)
    blades = bases(layout, mvClass)

    return layout, blades


def bases(layout, mvClass=MultiVector, grades=None):
    """Returns a dictionary mapping basis element names to their MultiVector
    instances, optionally for specific grades

    if you are lazy,  you might do this to populate your namespace
    with the variables of a given layout.

    >>> locals().update(layout.blades())

    bases(layout) --> {'name': baseElement, ...}
    """

    dict = {}
    for i in range(layout.gaDims):
        grade = layout.gradeList[i]
        if grade != 0:
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
    >>>randomMV(layout, min=-2.0, max=2.0, grades=None, uniform=None, n=2)

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
    """Makes repr(M) default to pretty-print.

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
    """Makes repr(M) default to eval-able representation.

    ugly()
    """

    global _pretty
    _pretty = False


def eps(newEps=None):
    """Get/Set the epsilon for float comparisons.

    eps(newEps)
    """

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


def gp(M, N):
    """
    Geometric product

    gp(M, N) =  M * N

    M and N must be from the same layout

    This is useful in calculating series of products, with `reduce()`
    for example

    >>>Ms = [M1, M2, M3] # list of multivectors
    >>>reduce(gp, Ms) #  == M1*M2*M3

    """

    return M*N


def ip(M, N):
    """
    Inner product function

    ip(M, N) =  M | N

    M and N must be from the same layout

    """

    return M ^ N


def op(M, N):
    """
    Outer product function

    op(M, N) =  M ^ N

    M and N must be from the same layout

    This is useful in calculating series of products, with `reduce()`
    for example

    >>>Ms = [M1, M2, M3] # list of multivectors
    >>>reduce(op, Ms) #  == M1^M2^M3

    """

    return M ^ N


def conformalize(layout, added_sig=[1, -1], **kw):
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
    **kw: kwargs
        passed to Cl() used to generate conformal layout

    Returns
    ---------
    layout_c:  `clifford.Layout`
        layout of the base GA
    blades_c: dict
        blades for the CGA
    stuff: dict
        dict containing the following:
            * ep - first basis vector added (usually positive)
            * en - second basis vector added (usually negative)
            * eo - zero vector of null basis (=.5*(en-ep))
            * einf - infinity vector of null basis (=en+ep)
            * E0 - minkowski bivector (=einf^eo)
            * base - pseudoscalar for base ga, in cga layout
            * up - up-project a vector from GA to CGA
            * down - down-project a vector from CGA to GA
            * homo - homogenize a CGA vector


    Examples
    ---------
    >>> from clifford import Cl, conformalize
    >>> G2, blades = Cl(2)
    >>> G2c, bladesc, stuff = conformalize(G2)
    >>> locals().update(bladesc)
    >>> locals().update(stuff)
    '''

    sig_c = list(layout.sig) + added_sig
    layout_c, blades_c = Cl(sig=sig_c, firstIdx=layout.firstIdx, **kw)
    basis_vectors = layout_c.basis_vectors
    added_keys = sorted(layout_c.basis_vectors.keys())[-2:]
    ep, en = [basis_vectors[k] for k in added_keys]

    # setup  null basis, and minkowski subspace bivector
    eo = .5 ^ (en - ep)
    einf = en + ep

    layout_c.isconformal = True
    layout_c.einf = einf
    layout_c.eo = eo

    E0 = einf ^ eo
    I_base = layout_c.pseudoScalar*E0

    # some convenience functions
    def up(x):
        try:
            if x.layout == layout:
                # vector is in original space, map it into conformal space
                old_val = x.value
                new_val = zeros(layout_c.gaDims)
                new_val[:len(old_val)] = old_val
                x = layout_c.MultiVector(value=new_val)
        except(AttributeError):
            # if x is a scalar it doesnt have layout but following
            # will still work
            pass

        # then up-project into a null vector
        return x + (.5 ^ ((x**2)*einf)) + eo

    def homo(x):
        return x*(-x | einf)(0).normalInv()  # homogenise conformal vector

    def down(x):
        x_down = (homo(x) ^ E0)*E0
        # new_val = x_down.value[:layout.gaDims]
        # create vector in layout (not cga)
        # x_down = layout.MultiVector(value=new_val)
        return x_down

    stuff = {}
    stuff.update({
        'ep': ep, 'en': en, 'eo': eo, 'einf': einf, 'E0': E0,
        'up': up, 'down': down, 'homo': homo, 'I_base': I_base
    })

    return layout_c, blades_c, stuff


# TODO: fix caching to work
# generate pre-defined algebras and cache them

# sigs = [(1, 1, 0), (2, 0, 0), (3, 1, 0), (3, 0, 0), (3, 2, 0), (4, 0, 0)]
# current_module = sys.modules[__name__]
# caching.build_or_read_cache_and_attach_submods(current_module, sigs=sigs)
