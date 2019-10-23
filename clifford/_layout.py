import re
from functools import reduce
from typing import List

import numpy as np
import numba
import sparse

# TODO: move some of these functions to this file if they're not useful anywhere
# else
import clifford as cf
from . import (
    get_adjoint_function,
    construct_tables,
    get_mult_function,
    get_leftLaInv,
    generate_blade_tup_map,
    generate_bitmap_to_linear_index_map,
    val_get_left_gmt_matrix,
    val_get_right_gmt_matrix,
)

from .io import read_ga_file


# The blade finding regex for parsing strings of mvs
_blade_pattern = re.compile(r"""
    ((^|\s)-?\s?\d+(\.\d+)?)\s|
    ((^|\+|-)\s?(\d+((e(\+|-))|\.)?(\d+)?)\^e\d+(\s|$))
""", re.VERBOSE)


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

    def parse_multivector(self, mv_string: str) -> 'cf.MultiVector':
        """ Parses a multivector string into a MultiVector object """
        # Get the names of the canonical blades
        blade_name_index_map = {name: index for index, name in enumerate(self.names)}

        # Clean up the input string a bit
        cleaned_string = re.sub('[()]', '', mv_string)

        # Create a multivector
        mv_out = cf.MultiVector(self)

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
        self.omt = sparse.where(omt_prod_mask, self.gmt, self.gmt.dtype.type(0))
        self.imt = sparse.where(imt_prod_mask, self.gmt, self.gmt.dtype.type(0))
        self.lcmt = sparse.where(lcmt_prod_mask, self.gmt, self.gmt.dtype.type(0))

        # This generates the functions that will perform the various products
        self.gmt_func = get_mult_function(self.gmt, self.gradeList)
        self.imt_func = get_mult_function(self.imt, self.gradeList)
        self.omt_func = get_mult_function(self.omt, self.gradeList)
        self.lcmt_func = get_mult_function(self.lcmt, self.gradeList)
        self.inv_func = get_leftLaInv(self.gmt, self.gradeList)

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
            self.gmt, self.gradeList,
            grades_a=grades_a, grades_b=grades_b, filter_mask=filter_mask
        )

    def omt_func_generator(self, grades_a=None, grades_b=None, filter_mask=None):
        return get_mult_function(
            self.gmt, self.gradeList,
            grades_a=grades_a, grades_b=grades_b, filter_mask=filter_mask
        )

    def lcmt_func_generator(self, grades_a=None, grades_b=None, filter_mask=None):
        return get_mult_function(
            self.gmt, self.gradeList,
            grades_a=grades_a, grades_b=grades_b, filter_mask=filter_mask
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
        return cf.MultiVector(layout=self, *args, **kw)

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
        return cf.randomMV(layout=self, **kw)

    def randomV(self, n=1, **kw):
        '''
        generate n random 1-vector s

        '''
        kw.update(dict(n=n, grades=[1]))
        return cf.randomMV(layout=self, **kw)

    def randomRotor(self):
        '''
        generate a random Rotor.

        this is created by muliplying an N unit vectors, where N is
        the dimension of the algebra if its even; else its one less.

        '''
        n = self.dims if self.dims % 2 == 0 else self.dims - 1
        R = reduce(cf.gp, self.randomV(n, normed=True))
        return R

    @property
    def basis_vectors(self):
        return cf.basis_vectors(self)

    @property
    def basis_vectors_lst(self):
        d = self.basis_vectors
        return [d[k] for k in sorted(d.keys())]

    def blades_of_grade(self, grade: int) -> List['MultiVector']:
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
        return [k for k in self.blades_list if k.grades() == {grade}]

    @property
    def blades_list(self) -> List['MultiVector']:
        '''
        List of blades in this layout matching the order of `self.bladeTupList`
        '''
        blades = self.blades
        return [blades[n] for n in self.names]

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
        return cf.bases(layout=self, *args, **kw)
