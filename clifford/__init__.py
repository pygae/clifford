"""

Two classes, Layout and MultiVector, and several helper functions are
provided to implement the algebras.


Classes
===============

.. autosummary::
    :toctree: generated/

    MultiVector
    Layout
    Frame

Helper Functions
================


.. autosummary::
    :toctree: generated/

    Cl
    conformalize
    bases
    randomMV
    pretty
    ugly
    eps





Issues
======

 * Currently, algebras over 6 dimensions are very slow. this is because
   this module was written for *pedagogical* purposes. However, because the
   syntax for this module is so attractive, we plan to fix the
   perfomance problems,  in the future...

 * Due to Python's order of operations, the bit operators ^ << follow
   the normal arithmetic operators + - * /, so

     1^e0 + 2^e1  !=  (1^e0) + (2^e1)

   as is probably intended.  Additionally,

     M = MultiVector(layout2D)  # null multivector
     M << 1^e0 << 2^e1 == 10.0^e1 + 1.0^e01
     M == 1.0
     e0 == 2 + 1^e0

   as is definitely not intended.  However,

     M = MultiVector(layout2D)
     M << (2^e0) << e1 << (3^e01) == M == 2^e0 + 1^e1 + 3^e01
     e0 == 1^e0
     e1 == 1^e1
     e01 == 1^e01

 * Since * is the inner product and the inner product with a scalar
   vanishes by definition, an expression like

     1|e0 + 2|e1

   is null.  Use the outer product or full geometric product, to
   multiply scalars with MultiVectors.  This can cause problems if
   one has code that mixes Python numbers and MultiVectors.  If the
   code multiplies two values that can each be either type without
   checking, one can run into problems as "1 | 2" has a very different
   result from the same multiplication with scalar MultiVectors.

 * Taking the inverse of a MultiVector will use a method proposed by
   Christian Perwass that involves the solution of a matrix equation.
   A description of that method follows:

   Representing multivectors as 2**dims vectors (in the matrix sense),
   we can carry out the geometric product with a multiplication table.
   In pseudo-tensorish language (using summation notation):

     m_i * g_ijk * n_k = v_j

   Suppose m_i are known (M is the vector we are taking the inverse of),
   the g_ijk have been computed for this algebra, and v_j = 1 if the
   j'th element is the scalar element and 0 otherwise, we can compute the
   dot product m_i * g_ijk.  This yields a rank-2 matrix.  We can
   then use well-established computational linear algebra techniques
   to solve this matrix equation for n_k.  The laInv method does precisely
   that.

   The usual, analytic, method for computing inverses [M**-1 = ~M/(M*~M) iff
   M*~M == |M|**2] fails for those multivectors where M*~M is not a scalar.
   It is only used if the inv method is manually set to point to normalInv.

   My testing suggests that laInv works.  In the cases where normalInv works,
   laInv returns the same result (within _eps).  In all cases,
   M * M.laInv() == 1.0 (within _eps).  Use whichever you feel comfortable
   with.

   Of course, a new issue arises with this method.  The inverses found
   are sometimes dependant on the order of multiplication.  That is:

     M.laInv() * M == 1.0
     M * M.laInv() != 1.0

   XXX Thus, there are two other methods defined, leftInv and rightInv which
   point to leftLaInv and rightLaInv.  The method inv points to rightInv.
   Should the user choose, leftInv and rightInv will both point to normalInv,
   which yields a left- and right-inverse that are the same should either exist
   (the proof is fairly simple).

 * The basis vectors of any algebra will be orthonormal unless you supply
   your own multiplication tables (which you are free to do after the Layout
   constructor is called).  A derived class could be made to calculate these
   tables for you (and include methods for generating reciprocal bases and the
   like).

 * No care is taken to preserve the dtype of the arrays.  The purpose
   of this module is pedagogical.  If your application requires so many
   multivectors that storage becomes important, the class structure here
   is unsuitable for you anyways.  Instead, use the algorithms from this
   module and implement application-specific data structures.

 * Conversely, explicit typecasting is rare.  MultiVectors will have
   integer coefficients if you instantiate them that way.  Dividing them
   by Python integers will have the same consequences as normal integer
   division.  Public outcry will convince me to add the explicit casts
   if this becomes a problem.


Acknowledgements
+++++++++++++++++
Konrad Hinsen fixed a few bugs in the conversion to numpy and adding some unit
tests.


ChangeLog
=========

Changes 0.6-0.7
+++++++++++++++++

 * Added a real license.
 * Convert to numpy instead of Numeric.

Changes 0.5-0.6
+++++++++++++++++

 * join() and meet() actually work now, but have numerical accuracy problems
 * added clean() to MultiVector
 * added leftInv() and rightInv() to MultiVector
 * moved pseudoScalar() and invPS() to MultiVector (so we can derive
   new classes from MultiVector)
 * changed all of the instances of creating a new MultiVector to create
   an instance of self.__class__ for proper inheritance
 * fixed bug in laInv()
 * fixed the massive confusion about how dot() works
 * added left-contraction
 * fixed embarassing bug in gmt generation
 * added normal() and anticommutator() methods
 * fixed dumb bug in elements() that limited it to 4 dimensions

Happy hacking!

Robert Kern

robert.kern@gmail.com
"""

from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
from past.builtins import cmp, range
from functools import reduce
import sys
import re

# Standard library imports.
import math
import numbers
import itertools
from warnings import warn

# Major library imports.
import numpy as np
from numpy import linalg, array
import numba



# The blade finding regex for parsing strings of mvs
_blade_pattern =  "((^|\s)-?\s?\d+(\.\d+)?)\s|(-\s?(\d+((e(\+|-))|\.)?(\d+)?)\^e\d+(\s|$))|((^|\+)\s?(\d+((e(\+|-))|\.)?(\d+)?)\^e\d+(\s|$))"
_eps = 1e-12            # float epsilon for float comparisons
_pretty = True          # pretty-print global
_print_precision = 5    # pretty printing precision on floats


def get_longest_string(string_array):
    """
    Return the longest string in a list of strings
    """
    return max(string_array,key=len)

def _sign(seq, orig):
    """Determine {even,odd}-ness of permutation seq or orig.

    Returns 1 if even; -1 if odd.
    """

    sign = 1
    seq = list(seq)

    for i in range(len(seq)):
            if seq[i] != orig[i]:
                j = seq.index(orig[i])
                sign = -sign
                seq[i], seq[j] = seq[j], seq[i]
    return sign

@numba.njit
def _containsDups(input_list):
    """
    Simply checks if the input list contains duplicates
    """
    for k in input_list:
        if input_list.count(k) != 1:
            return 1
    return 0

@numba.njit
def modify_idx(idx, grade):
    """
    This function is called during the even/odd grade algorithm
    It is jitted to make it as fast as possible
    """
    j = grade - 1
    done = 0
    while not done:
        idx[j] = idx[j] + 1
        while idx[j] == grade:
            idx[j] = 0
            j = j - 1
            idx[j] = idx[j] + 1
            if j == -1:
                raise NoMorePermutations()
        j = grade - 1
        if not _containsDups(idx):
            done = 1

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


def get_mult_function(mult_table,n_dims,gradeList,grades_a=None,grades_b=None,filter_mask=None):
    ''' 
    Returns a function that implements the mult_table on two input multivectors
    '''
    non_zero_indices = mult_table.nonzero()
    k_list = non_zero_indices[0]
    l_list = non_zero_indices[1]
    m_list = non_zero_indices[2]
    mult_table_vals = np.array([mult_table[k,l,m] for k,l,m in np.transpose(non_zero_indices)],dtype=int)
    
    if filter_mask is not None:
        # We can pass the sparse filter mask directly
        k_list = k_list[filter_mask]
        l_list = l_list[filter_mask]
        m_list = m_list[filter_mask]
        mult_table_vals = mult_table_vals[filter_mask]
        @numba.njit
        def mv_mult(value,other_value):
            output = np.zeros(n_dims)
            for ind,k in enumerate(k_list):
                m = m_list[ind]
                l = l_list[ind]
                output[l] += value[k]*mult_table_vals[ind]*other_value[m]
            return output
        return mv_mult
        
    elif ((grades_a is not None) and (grades_b is not None)):
        # We can also specify sparseness by grade
        filter_mask = np.zeros(len(k_list), dtype=bool)
        for i in range(len(filter_mask)):
            if gradeList[k_list[i]] in grades_a:
                if gradeList[m_list[i]] in grades_b:
                    filter_mask[i] = 1

        k_list = k_list[filter_mask]
        l_list = l_list[filter_mask]
        m_list = m_list[filter_mask]
        mult_table_vals = mult_table_vals[filter_mask]
        
        @numba.njit
        def mv_mult(value,other_value):
            output = np.zeros(n_dims)
            for ind,k in enumerate(k_list):
                m = m_list[ind]
                l = l_list[ind]
                output[l] += value[k]*mult_table_vals[ind]*other_value[m]
            return output
        return mv_mult

    # This case we specify no sparseness in advance, the algorithm checks for zeros
    @numba.njit
    def mv_mult(value,other_value):
        output = np.zeros(n_dims)
        for ind,k in enumerate(k_list):
            v_val = value[k]
            if v_val!=0.0:
                m = m_list[ind]
                ov_val = other_value[m]
                if ov_val!=0.0:
                    l = l_list[ind]
                    output[l] += v_val*mult_table_vals[ind]*ov_val
        return output
    return mv_mult


def _myDot(a, b):
    """Returns the inner product as *I* learned it.

    a_i...k * b_k...m = c_i...m     in summation notation with the ...'s
                                    representing arbitrary, omitted indices

    The sum is over the last axis of the first argument and the first axis
    of the last axis.

    _myDot(a, b) --> NumPy array
    """

    a = np.asarray(a)
    b = np.asarray(b)

    tempAxes = tuple(list(range(1, len(b.shape))) + [0])
    newB = np.transpose(b, tempAxes)

    # innerproduct sums over the *last* axes of *both* arguments
    return np.inner(a, newB)


class NoMorePermutations(Exception):
    """ No more permutations can be generated.
    """


class Layout(object):
    """ Layout stores information regarding the geometric algebra itself and the
    internal representation of multivectors.

    It is constructed like this:

      Layout(signature, bladeTupList, firstIdx=0, names=None)

    The arguments to be passed are interpreted as follows:

      signature -- the signature of the vector space.  This should be
          a list of positive and negative numbers where the sign determines the
          sign of the inner product of the corresponding vector with itself.
          The values are irrelevant except for sign.  This list also determines
          the dimensionality of the vectors.  Signatures with zeroes are not
          permitted at this time.

          Examples:
            signature = [+1, -1, -1, -1] # Hestenes', et al. Space-Time Algebra
            signature = [+1, +1, +1]     # 3-D Euclidean signature

      bladeTupList -- list of tuples corresponding to the blades in the whole
          algebra.  This list determines the order of coefficients in the
          internal representation of multivectors.  The entry for the scalar
          must be an empty tuple, and the entries for grade-1 vectors must be
          singleton tuples.  Remember, the length of the list will be 2**dims.

          Example:
            bladeTupList = [(), (0,), (1,), (0,1)]  # 2-D

      firstIdx -- the index of the first vector.  That is, some systems number
          the base vectors starting with 0, some with 1.  Choose by passing
          the correct number as firstIdx.  0 is the default.

      names -- list of names of each blade.  When pretty-printing multivectors,
          use these symbols for the blades.  names should be in the same order
          as bladeTupList.  You may use an empty string for scalars.  By
          default, the name for each non-scalar blade is 'e' plus the indices
          of the blade as given in bladeTupList.

          Example:
            names = ['', 's0', 's1', 'i']  # 2-D


    Layout's Members:

      dims -- dimensionality of vectors (== len(signature))

      sig -- normalized signature (i.e. all values are +1 or -1)

      firstIdx -- starting point for vector indices

      bladeTupList -- list of blades

      gradeList -- corresponding list of the grades of each blade

      gaDims -- 2**dims

      names -- pretty-printing symbols for the blades

      even -- dictionary of even permutations of blades to the canonical blades

      odd -- dictionary of odd permutations of blades to the canonical blades

      gmt -- multiplication table for geometric product [1]

      imt -- multiplication table for inner product [1]

      omt -- multiplication table for outer product [1]

      lcmt -- multiplication table for the left-contraction [1]


    [1] The multiplication tables are NumPy arrays of rank 3 with indices like
        the tensor g_ijk discussed above.
    """

    def __init__(self, sig, bladeTupList, firstIdx=0, names=None):
        self.dims = len(sig)
        self.sig = np.divide(sig, np.absolute(sig)).astype(int)
        self.firstIdx = firstIdx

        self.bladeTupList = list(map(tuple, bladeTupList))
        self._checkList()

        self.gaDims = len(self.bladeTupList)
        self.gradeList = list(map(len, self.bladeTupList))

        if names is None or isinstance(names, str):
            if isinstance(names, str):
                e = names
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

        self._genEvenOdd()
        self._genTables()
        self.adjoint_func = get_adjoint_function(self.gradeList)

    def __repr__(self):
        s = ("Layout(%r, %r, firstIdx=%r, names=%r)" % (
                list(self.sig),
                self.bladeTupList, self.firstIdx, self.names))
        return s

    def __eq__(self,other):
        if other is not self:
            return np.array_equal(self.sig,other.sig)
        else:
            return True

    def __ne__(self,other):
        if other is self:
            return False
        else:
            return not np.array_equal(self.sig,other.sig)

    def _genEvenOdd(self):
        "Make mappings of even and odd permutations to their canonical blades."

        self.even = {}
        self.odd = {}

        for i in range(self.gaDims):
            blade = self.bladeTupList[i]
            grade = self.gradeList[i]

            if grade in (0, 1):
                # handled easy cases
                self.even[blade] = blade
                continue
            elif grade == 2:
                # another easy case
                self.even[blade] = blade
                self.odd[(blade[1], blade[0])] = blade
                continue
            else:
                # general case, lifted from Chooser.py released on
                # comp.lang.python by James Lehmann with permission.
                idx = list(range(grade))
                try:
                    for i in range(np.multiply.reduce(range(1, grade+1))):
                        # grade! permutations

                        # Whatever this does
                        modify_idx(idx,grade)
                        perm = tuple([blade[k] for k in idx])

                        if _sign(perm, blade) == 1:
                            self.even[perm] = blade
                        else:
                            self.odd[perm] = blade

                except NoMorePermutations:
                    pass

                self.even[blade] = blade

    def parse_multivector(self,mv_string):
        # Get the names of the canonical blades
        blade_name_index_map = {name:index for index,name in enumerate(self.names)}

        # Clean up the input string a bit
        cleaned_string = re.sub('[()]','',mv_string)

        # Apply the regex
        search_result = re.findall(_blade_pattern,cleaned_string)
        
        # Create a multivector
        mv_out = MultiVector(self)
        for res in search_result:
            # Clean up the search result
            cleaned_match = get_longest_string(res)

            # Split on the '^'
            stuff = cleaned_match.split('^')

            if (len(stuff) == 2):
                # Extract the value of the blade and the index of the blade
                blade_val = float("".join(stuff[0].split()))
                blade_index = blade_name_index_map[stuff[1].strip()]
                mv_out[blade_index] = blade_val
            else:
                if(len(stuff) == 1):
                    # Extract the value of the scalar
                    blade_val = float("".join(stuff[0].split()))
                    blade_index = 0;
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

    def _gmtElement(self, a, b):
        "Element of the geometric multiplication table given blades a, b."

        mul = 1         # multiplier

        newBlade = list(a) + list(b)

        unique = 0
        while unique == 0:
            for i in range(len(newBlade)):
                index = newBlade[i]
                if newBlade.count(index) != 1:
                    delta = newBlade[i+1:].index(index)
                    eo = 1 - 2*(delta % 2)
                    # eo == 1 if the elements are an even number of flips away
                    # eo == -1 otherwise

                    del newBlade[i+delta+1]
                    del newBlade[i]

                    mul = eo * mul * self.sig[index - self.firstIdx]
                    break
            else:
                unique = 1

        newBlade = tuple(newBlade)

        if newBlade in self.bladeTupList:
            idx = self.bladeTupList.index(newBlade)
            # index of the product blade in the bladeTupList
        elif newBlade in self.even.keys():
            # even permutation
            idx = self.bladeTupList.index(self.even[newBlade])
        else:
            # odd permutation
            idx = self.bladeTupList.index(self.odd[newBlade])
            mul = -mul

        element = np.zeros((self.gaDims,), dtype=int)
        element[idx] = mul

        return element, idx

    def _genTables(self):
        "Generate the multiplication tables."

        # geometric multiplication table
        gmt = np.zeros((self.gaDims, self.gaDims, self.gaDims), dtype=int)
        # inner product table
        imt = np.zeros((self.gaDims, self.gaDims, self.gaDims), dtype=int)
        # outer product table
        omt = np.zeros((self.gaDims, self.gaDims, self.gaDims), dtype=int)
        # left-contraction table
        lcmt = np.zeros((self.gaDims, self.gaDims, self.gaDims), dtype=int)

        for i in range(self.gaDims):
            for j in range(self.gaDims):
                gmt[i, :, j], idx = self._gmtElement(
                        list(self.bladeTupList[i]), list(self.bladeTupList[j]))

                if (
                        self.gradeList[idx] == abs(
                            self.gradeList[i] - self.gradeList[j]) and
                        self.gradeList[i] != 0 and self.gradeList[j] != 0):

                    # A_r . B_s = <A_r B_s>_|r-s|
                    # if r,s != 0
                    imt[i, :, j] = gmt[i, :, j]

                if self.gradeList[idx] == (
                        self.gradeList[i] + self.gradeList[j]):

                    # A_r ^ B_s = <A_r B_s>_|r+s|
                    omt[i, :, j] = gmt[i, :, j]

                if self.gradeList[idx] == (
                        self.gradeList[j] - self.gradeList[i]):
                    # A_r _| B_s = <A_r B_s>_(s-r) if s-r >= 0
                    lcmt[i, :, j] = gmt[i, :, j]

        self.gmt_func = get_mult_function(gmt,self.gaDims,self.gradeList)
        self.imt_func = get_mult_function(imt,self.gaDims,self.gradeList)
        self.omt_func = get_mult_function(omt,self.gaDims,self.gradeList)
        self.lcmt_func = get_mult_function(lcmt,self.gaDims,self.gradeList)
        self.gmt = gmt
        self.imt = imt
        self.omt = omt
        self.lcmt = lcmt

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
    
    @property
    def pseudoScalar(self):
        return self.blades_list[-1]

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

    def __init__(self, layout, value=None, string=None):
        """Constructor.

        MultiVector(layout, value=None) --> MultiVector
        """

        self.layout = layout

        if value is None:
            if string is None:
                self.value = np.zeros((self.layout.gaDims,), dtype=float)
            else:
                self.value = layout.parse_multivector(string).value
        else:
            self.value = np.array(value)
            if self.value.shape != (self.layout.gaDims,):
                raise ValueError(
                    "value must be a sequence of length %s" %
                    self.layout.gaDims)

    def __array_wrap__(self, out_arr, context=None):
        '''
        This is a work-around needed to prevent numpy arrays from
        operating  on MultiVectors using their operators.  This happens
        because Multivectors act enough like an array, that duck
        typing thinks they are an array.
        '''
        
        uf, objs, out_index = context
        if uf.__name__ =='multiply':
            return objs[1]*objs[0]
        if uf.__name__ == 'divide':
            return objs[1].inv()*objs[0]
        elif uf.__name__ == 'add':
            return objs[1] + objs[0]
        elif uf.__name__ == 'subtract':
            return -objs[1] + objs[0]
        elif uf.__name__ == 'exp':
            return math.e**(objs[0])

        else:
            raise ValueError('i dont know how to %s yet' % uf.__name__)

    def _checkOther(self, other, coerce=1):
        """Ensure that the other argument has the same Layout or coerce value if
        necessary/requested.

        _checkOther(other, coerce=1) --> newOther, isMultiVector
        """
        if isinstance(other, numbers.Number):
            if coerce:
                # numeric scalar
                newOther = self._newMV()
                newOther[()] = other
                return newOther, True
            else:
                return other, False

        elif (
                isinstance(other, self.__class__) and
                other.layout != self.layout):
            raise ValueError(
                "cannot operate on MultiVectors with different Layouts")

        return other, True

    def _newMV(self, newValue=None):
        """Returns a new MultiVector (or derived class instance).

        _newMV(self, newValue=None)
        """

        return self.__class__(self.layout, newValue)

    # numeric special methods
    # binary

    def __mul__(self, other):
        """Geometric product

        M * N --> MN
        __and__(other) --> MultiVector
        """

        other, mv = self._checkOther(other, coerce=0)

        if mv:
            newValue = self.layout.gmt_func(self.value,other.value)
        else:
            newValue = other * self.value

        return self._newMV(newValue)

    def __rmul__(self, other):
        """Right-hand geometric product

        N * M --> NM
        __rand__(other) --> MultiVector
        """

        other, mv = self._checkOther(other, coerce=0)

        if mv:
            newValue = self.layout.gmt_func(other.value,self.value)
        else:
            newValue = other*self.value

        return self._newMV(newValue)

    def __xor__(self, other):
        """Outer product

        M ^ N
        __xor__(other) --> MultiVector
        """

        other, mv = self._checkOther(other, coerce=0)

        if mv:
            newValue = self.layout.omt_func(self.value,other.value)
        else:
            newValue = other*self.value

        return self._newMV(newValue)

    def __rxor__(self, other):
        """Right-hand outer product

        N ^ M
        __rxor__(other) --> MultiVector
        """

        other, mv = self._checkOther(other, coerce=0)

        if mv:
            newValue = self.layout.omt_func(other.value,self.value)
        else:
            newValue = other * self.value

        return self._newMV(newValue)

    def __or__(self, other):
        """Inner product

        M | N
        __mul__(other) --> MultiVector
        """

        other, mv = self._checkOther(other)

        if mv:
            newValue = self.layout.imt_func(self.value,other.value)
        else:
            return self._newMV()  # l * M = M * l = 0 for scalar l

        return self._newMV(newValue)

    __ror__ = __or__

    def __add__(self, other):
        """Addition

        M + N
        __add__(other) --> MultiVector
        """

        other, mv = self._checkOther(other)
        newValue = self.value + other.value

        return self._newMV(newValue)

    __radd__ = __add__

    def __sub__(self, other):
        """Subtraction

        M - N
        __sub__(other) --> MultiVector
        """

        other, mv = self._checkOther(other)
        newValue = self.value - other.value

        return self._newMV(newValue)

    def __rsub__(self, other):
        """Right-hand subtraction

        N - M
        __rsub__(other) --> MultiVector
        """

        other, mv = self._checkOther(other)
        newValue = other.value - self.value

        return self._newMV(newValue)

    def __truediv__(self, other):
        """Division
                       -1
        M / N --> M * N
        __div__(other) --> MultiVector
        """

        other, mv = self._checkOther(other, coerce=0)

        if mv:
            return self * other.inv()
        else:
            newValue = self.value / other
            return self._newMV(newValue)

    def __rtruediv__(self, other):
        """Right-hand division
                       -1
        N / M --> N * M
        __rdiv__(other) --> MultiVector
        """

        other, mv = self._checkOther(other)

        return other * self.inv()

    if sys.version_info[0] < 3:
        __div__ = __truediv__
        __rdiv__ = __rtruediv__

    def __pow__(self, other):
        """Exponentiation of a multivector by an integer
                    n
        M ** n --> M
        __pow__(other) --> MultiVector
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

    def __rpow__(self, other):
        """Exponentiation of a real by a multivector
                  M
        r**M --> r
        __rpow__(other) --> MultiVector
        """

        # Let math.log() check that other is a Python number, not something
        # else.
        intMV = math.log(other) * self
        # pow(x, y) == exp(y * log(x))

        newMV = self._newMV()  # null

        nextTerm = self._newMV()
        nextTerm[()] = 1  # order 0 term of exp(x) Taylor expansion

        n = 1.

        while nextTerm != 0:
            # iterate until the added term is within _eps of 0
            newMV << nextTerm
            nextTerm = nextTerm * intMV / n
            n = n + 1
        else:
            # squeeze out that extra little bit of accuracy
            newMV << nextTerm

        return newMV

    def __lshift__(self, other):
        """In-place addition

        M << N --> M + N
        __iadd__(other) --> MultiVector
        """

        other, mv = self._checkOther(other)

        self.value = self.value + other.value

        return self

    # unary

    def __neg__(self):
        """Negation

        -M
        __neg__() --> MultiVector
        """

        newValue = -self.value

        return self._newMV(newValue)

    def __pos__(self):
        """Positive (just a copy)

        +M
        __pos__(self) --> MultiVector
        """

        newValue = self.value + 0  # copy

        return self._newMV(newValue)

    def mag2(self):
        """Magnitude (modulus) squared
           2
        |M|
        mag2() --> PyFloat | PyInt

        Note in mixed signature spaces this may be negative
        """

        return (~self * self)[()]

    def __abs__(self):
        """Magnitude (modulus)

        abs(M) --> |M|
        __abs__() --> PyFloat

        This is sqrt(abs(~M*M)).

        The abs inside the sqrt is need for spaces of mixed signature
        """

        return np.sqrt(abs(self.mag2()))

    def adjoint(self):
        """Adjoint / reversion
               _
        ~M --> M (any one of several conflicting notations)
        ~(N * M) --> ~M * ~N
        adjoint() --> MultiVector
        """
        # The multivector created by reversing all multiplications
        return self._newMV(self.layout.adjoint_func(self.value))

    __invert__ = adjoint

    # builtin
    def __int__(self):
        """Coerce to an integer iff scalar.

        int(M)
        __int__() --> PyInt
        """

        return int(self.__float__())

    if sys.version_info[0] < 3:
        def __long__(self):
            """Coerce to a long iff scalar.

            long(M)
            __long__() --> PyLong
            """

            return long(self.__float__())

    def __float__(self):
        """"Coerce to a float iff scalar.

        float(M)
        __float__() --> PyFloat
        """

        if self.isScalar():
            return float(self[()])
        else:
            raise ValueError("non-scalar coefficients are non-zero")

    # sequence special methods
    def __len__(self):
        """Returns length of value array.

        len(M)
        __len__() --> PyInt
        """

        return self.layout.gaDims

    def __getitem__(self, key):
        """If key is a blade tuple (e.g. (0,1) or (1,3)), then return
        the (real) value of that blade's coefficient.
        Otherwise, treat key as an index into the list of coefficients.

        M[blade] --> PyFloat | PyInt
        M[index] --> PyFloat | PyInt
        __getitem__(key) --> PyFloat | PyInt
        """

        if key in self.layout.bladeTupList:
            return self.value[self.layout.bladeTupList.index(key)]
        elif key in self.layout.even:
            return self.value[self.layout.bladeTupList.index(
                self.layout.even[key])]
        elif key in self.layout.odd:
            return -self.value[self.layout.bladeTupList.index(
                self.layout.odd[key])]
        else:
            return self.value[key]

    def __setitem__(self, key, value):
        """If key is a blade tuple (e.g. (0,1) or (1,3)), then set
        the (real) value of that blade's coeficient.
        Otherwise treat key as an index into the list of coefficients.

        M[blade] = PyFloat | PyInt
        M[index] = PyFloat | PyInt
        __setitem__(key, value)
        """

        if key in self.layout.bladeTupList:
            self.value[self.layout.bladeTupList.index(key)] = value
        elif key in self.layout.even:
            self.value[self.layout.bladeTupList.index(
                self.layout.even[key])] = value
        elif key in self.layout.odd:
            self.value[self.layout.bladeTupList.index(
                self.layout.odd[key])] = -value
        else:
            self.value[key] = value

    def __delitem__(self, key):
        """Set the selected coefficient to 0.

        del M[blade]
        del M[index]
        __delitem__(key)
        """

        if key in self.layout.bladeTupList:
            self.value[self.layout.bladeTupList.index(key)] = 0
        elif key in self.layout.even:
            self.value[self.layout.bladeTupList.index(
                self.layout.even[key])] = 0
        elif key in self.layout.odd:
            self.value[self.layout.bladeTupList.index(
                self.layout.odd[key])] = 0
        else:
            self.value[key] = 0

    def __getslice__(self, i, j):
        """Return a copy with only the slice non-zero.

        M[i:j]
        __getslice__(i, j) --> MultiVector
        """

        newMV = self._newMV()
        newMV.value[i:j] = self.value[i:j]

        return newMV

    def __setslice__(self, i, j, sequence):
        """Paste a sequence into coefficients array.

        M[i:j] = sequence
        __setslice__(i, j, sequence)
        """

        self.value[i:j] = sequence

    def __delslice__(self, i, j):
        """Set slice to zeros.

        del M[i:j]
        __delslice__(i, j)
        """

        self.value[i:j] = 0

    # grade projection
    def __call__(self, other,*others):
        """Return a new multi-vector projected onto a grade OR a MV


        M(grade[s]) --> <M>
                        grade
        OR

        M(other) --> other.project(M)

        __call__(grade) --> MultiVector
        
        Examples
        --------
        >>>M(0)
        >>>M(0,2)
        """
        if isinstance(other, MultiVector):
            return other.project(self)
        else:
            # we are making a grade projection
            grade = other

        
        if len(others) !=0:
            return sum([self.__call__(k) for k in (other,)+others])
        
                
        if grade not in self.layout.gradeList:
            raise ValueError("algebra does not have grade %s" % grade)

        if not isinstance(grade, int):
            raise ValueError("grade must be an integer")

        mask = np.equal(grade, self.layout.gradeList)

        newValue = np.multiply(mask, self.value)

        return self._newMV(newValue)

    # fundamental special methods
    def __str__(self):
        """Return pretty-printed representation.

        str(M)
        __str__() --> PyString
        """

        s = ''
        p = _print_precision

        for i in range(self.layout.gaDims):
            # if we have nothing yet, don't use + and - as operators but
            # use - as an unary prefix if necessary
            if s:
                seps = (' + ', ' - ')
            else:
                seps = ('', '-')

            if self.layout.gradeList[i] == 0:
                # scalar
                if abs(self.value[i]) >= _eps:
                    if self.value[i] > 0:
                        s = '%s%s%s' % (s, seps[0], round(self.value[i], p))
                    else:
                        s = '%s%s%s' % (s, seps[1], -round(self.value[i], p))

            else:
                if abs(self.value[i]) >= _eps:
                    # not a scalar
                    if self.value[i] > 0:
                        s = '%s%s(%s^%s)' % (
                            s, seps[0], round(self.value[i], p),
                            self.layout.names[i])
                    else:
                        s = '%s%s(%s^%s)' % (
                            s, seps[1], -round(self.value[i], p),
                            self.layout.names[i])
        if s:
            # non-zero
            return s
        else:
            # return scalar 0
            return '0'

    def __repr__(self):
        """Return eval-able representation if global _pretty is false.
        Otherwise, return str(self).

        repr(M)
        __repr__() --> PyString
        """

        if _pretty:
            return self.__str__()

        s = "MultiVector(%s, value=%s)" % (
            repr(self.layout), list(self.value))
        return s

    def __bool__(self):
        """Instance is nonzero iff at least one of the coefficients
        is nonzero.

        __nonzero() --> Boolean
        """

        nonzeroes = np.absolute(self.value) > _eps

        if nonzeroes.any():
            return True
        else:
            return False

    if sys.version_info[0] < 3:
        __nonzero__ = __bool__

    if sys.version_info[0] < 3:
        def __cmp__(self, other):
            """Compares two multivectors.

            This is mostly defined for equality testing (within epsilon).
            In the case that the two multivectors have different Layouts,
            we will raise an error.  In the case that they are not equal,
            we will compare the tuple represenations of the coefficients
            lists just so as to return something valid.  Therefore,
            inequalities are well-nigh meaningless (since they are
            meaningless for multivectors while equality is meaningful).

            TODO: rich comparisons.

            M == N
            __cmp__(other) --> -1|0|1
            """

            other, mv = self._checkOther(other)
            print('cmp1', self.value)
            print('cmp2', other.value)

            if (np.absolute(self.value - other.value) < _eps).all():
                # equal within epsilon
                return 0
            else:
                return cmp(tuple(self.value), tuple(other.value))
    else:
        def __eq__(self, other):
            other, mv = self._checkOther(other)

            if (np.absolute(self.value - other.value) < _eps).all():
                # equal within epsilon
                return True
            else:
                return False

        def __ne__(self, other):
            return not self.__eq__(other)

        def __lt__(self, other):
            raise NotImplementedError

        def __le__(self, other):
            raise NotImplementedError

        def __gt__(self, other):
            raise NotImplementedError

        def __ge__(self, other):
            raise NotImplementedError

    def clean(self, eps=None):
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

    def round(self, eps=None):
        """Rounds all coefficients according to Python's rounding rules.

        eps defaults to the current value of the global _eps.

        round(eps=None)
        """

        if eps is None:
            eps = _eps

        self.value = np.around(self.value, eps)

        return self

    # Geometric Algebraic functions
    def lc(self, other):
        """Returns the left-contraction of two multivectors.

        M _| N
        lc(other) --> MultiVector
        """

        other, mv = self._checkOther(other, coerce=1)

        newValue = self.layout.lcmt_func(self.value,other.value)

        return self._newMV(newValue)

    def pseudoScalar(self):
        "Returns a MultiVector that is the pseudoscalar of this space."
        return self.layout.pseudoScalar
        

    def invPS(self):
        "Returns the inverse of the pseudoscalar of the algebra."

        ps = self.pseudoScalar()

        return ps.inv()

    def isScalar(self):
        """Returns true iff self is a scalar.

        isScalar() --> Boolean
        """

        indices = list(range(self.layout.gaDims))
        indices.remove(self.layout.gradeList.index(0))

        for i in indices:
            if abs(self.value[i]) < _eps:
                continue
            else:
                return False

        return True

    def isBlade(self):
        """Returns true if multivector is a blade.

        FIXME: Apparently, not all single-grade multivectors are blades. E.g.
        in the 4-D Euclidean space, a=(e1^e2 + e3^e4) is not a blade. There is
        no vector v such that v^a=0.

        isBlade() --> Boolean
        """

        grade = None

        for i in range(self.layout.gaDims):
            if abs(self.value[i]) > _eps:
                if grade is None:
                    grade = self.layout.gradeList[i]
                elif self.layout.gradeList[i] != grade:
                    return 0

        return 1

    def grades(self):
        """Return the grades contained in the multivector.

        grades() --> [ PyInt, PyInt, ... ]
        """

        grades = []

        for i in range(self.layout.gaDims):
            if abs(self.value[i]) > _eps and \
               self.layout.gradeList[i] not in grades:
                grades.append(self.layout.gradeList[i])

        return grades

    @property
    def blades_list(self):
        '''
        ordered list of blades present in this MV
        '''
        blades_list = self.layout.blades_list
        value = self.value

        b = [value[0]] + [value[k]*blades_list[k] for k in range(1, len(self))]
        return [k for k in b if k != 0]

    def normal(self):
        """Return the (mostly) normalized multivector.

        The _mostly_ comes from the fact that some multivectors have a
        negative squared-magnitude.  So, without introducing formally
        imaginary numbers, we can only fix the normalized multivector's
        magnitude to +-1.

        M / |M|  up to a sign
        normal() --> MultiVector
        """

        return self / np.sqrt(abs(self.mag2()))

    def leftLaInv(self):
        """Return left-inverse using a computational linear algebra method
        proposed by Christian Perwass.
         -1         -1
        M    where M  * M  == 1
        leftLaInv() --> MultiVector
        """

        identity = np.zeros((self.layout.gaDims,))
        identity[self.layout.gradeList.index(0)] = 1

        intermed = np.dot(self.layout.gmt, self.value)
        intermed = np.transpose(intermed)

        if abs(linalg.det(intermed)) < _eps:
            raise ValueError("multivector has no left-inverse")

        sol = linalg.solve(intermed, identity)

        return self._newMV(sol)

    def rightLaInv(self):
        """Return right-inverse using a computational linear algebra method
        proposed by Christian Perwass.
         -1              -1
        M    where M * M  == 1
        rightLaInv() --> MultiVector
        """

        identity = np.zeros((self.layout.gaDims,))
        identity[self.layout.gradeList.index(0)] = 1

        intermed = _myDot(self.value, self.layout.gmt)

        if abs(linalg.det(intermed)) < _eps:
            warn('Inverse might be inaccurate')
            #raise ValueError("multivector has no right-inverse")

        sol = linalg.solve(intermed, identity)

        return self._newMV(sol)

    def normalInv(self):
        """Returns the inverse of itself if M*~M == |M|**2.
         -1
        M   = ~M / (M * ~M)
        normalInv() --> MultiVector
        """

        Madjoint = ~self
        MadjointM = (Madjoint * self)

        if MadjointM.isScalar() and abs(MadjointM[()]) > _eps:
            # inverse exists
            return Madjoint / MadjointM[()]
        else:
            raise ValueError("no inverse exists for this multivector")

    leftInv = leftLaInv
    inv = rightInv = rightLaInv
    # inv= normalInv

    def dual(self, I=None):
        """Returns the dual of the multivector against the given subspace I.
        I defaults to the pseudoscalar.

        ~        -1
        M = M * I
        dual(I=None) --> MultiVector
        """

        if I is None:
            Iinv = self.invPS()
        else:
            Iinv = I.inv()

        return self * Iinv

    def commutator(self, other):
        """Returns the commutator product of two multivectors.

        [M, N] = M X N = (M*N - N*M)/2
        commutator(other) --> MultiVector
        """

        return ((self * other) - (other * self)) / 2

    x = commutator

    def anticommutator(self, other):
        """Returns the anti-commutator product of two multivectors.

        (M*N + N*M)/2
        anticommutator(other) --> MultiVector
        """

        return ((self * other) + (other * self)) / 2

    def gradeInvol(self):
        """Returns the grade involution of the multivector.
         *                    i
        M  = Sum[i, dims, (-1)  <M>  ]
                                   i
        gradeInvol() --> MultiVector
        """

        signs = np.power(-1, self.layout.gradeList)

        newValue = signs * self.value

        return self._newMV(newValue)

    @property
    def even(self):
        '''
        Even part of this mulivector

        defined as
        M + M.gradInvol()
        '''
        return .5*(self + self.gradeInvol())

    @property
    def odd(self):
        '''
        Odd part of this mulivector

        defined as
        M +- M.gradInvol()
        '''
        return .5*(self - self.gradeInvol())

    def conjugate(self):
        """Returns the Clifford conjugate (reversion and grade involution).
         *
        M  --> (~M).gradeInvol()
        conjugate() --> MultiVector
        """

        return (~self).gradeInvol()

    # Subspace operations
    def project(self, other):
        """Projects the multivector onto the subspace represented by this blade.
                            -1
        P (M) = (M _| A) * A
         A
        project(M) --> MultiVector
        """

        other, mv = self._checkOther(other, coerce=1)

        if not self.isBlade():
            raise ValueError("self is not a blade")

        return other.lc(self) * self.inv()

    def basis(self):
        """Finds a vector basis of this subspace.

        basis() --> [ MultiVector, MultiVector, ... ]
        """

        gr = self.grades()

        if len(gr) != 1:
            # FIXME: this is not a sufficient condition for a blade.
            raise ValueError("self is not a blade")

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

    def join(self, other):
        """Returns the join of two blades.
              .
        J = A ^ B
        join(other) --> MultiVector
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

    def meet(self, other, subspace=None):
        """Returns the meet of two blades.

        Computation is done with respect to a subspace that defaults to
        the join if none is given.
                     -1
        M \/i N = (Mi  ) * N
        meet(other, subspace=None) --> MultiVector
        """

        other, mv = self._checkOther(other)

        r = self.grades()
        s = other.grades()

        if len(r) > 1 or len(s) > 1:
            raise ValueError("not blades")

        if subspace is None:
            subspace = self.join(other)

        return (self * subspace.inv()) | other


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

    def is_innermorphic_to(self, other,eps=None):
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
            eps=_eps
            
            
        return array([float((b[m]|b[n]) - (a[m]|a[n]))<eps for m, n in pairs]).all()


class BladeMap(object):
    '''
    A Map Relating Blades in two different algebras

    Examples
    -----------
    >>> from clifford import Cl
    >>> # Dirac Algebra  `D`
    >>> D, D_blades = Cl(1,3, firstIdx=0, names='d')
    >>> locals().update(D_blades)

    >>> # Pauli Algebra  `P`
    >>> P, P_blades = Cl(3, names='p')
    >>> locals().update(P_blades)
    >>> sta_split = BladeMap([(d01,p1),
                              (d02,p2),
                              (d03,p3),
                              (d12,p12),
                              (d23,p23),
                              (d13,p13)])

    '''
    def __init__(self, blades_map, map_scalars=True):
        self.blades_map = blades_map

        if map_scalars:
            # make scalars in each algebra map
            s1 = self.b1[0]._newMV()+1
            s2 = self.b2[0]._newMV()+1
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
        B = to_b[0]._newMV()
        for from_obj, to_obj in zip(from_b, to_b):
            B += (sum(A.value*from_obj.value)*to_obj)
        return B


def comb(n, k):
    """\
    Returns /n\\
            \\k/

    comb(n, k) --> PyInt
    """

    def fact(n):
        if n == 0:
            return 1
        return np.multiply.reduce(range(1, n+1))

    return int(fact(n)/(fact(k)*fact(n - k)))


def elements(dims, firstIdx=0):
    """Return a list of tuples representing all 2**dims of blades
    in a dims-dimensional GA.

    elements(dims, firstIdx=0) --> bladeTupList
    """

    indcs = list(range(firstIdx, firstIdx + dims))

    blades = [()]

    for k in range(1, dims+1):
        # k = grade

        if k == 1:
            for i in indcs:
                blades.append((i,))
            continue

        curBladeX = indcs[:k]

        for i in range(comb(dims, k)):
            if curBladeX[-1] < firstIdx+dims-1:
                # increment last index
                blades.append(tuple(curBladeX))
                curBladeX[-1] = curBladeX[-1] + 1

            else:
                marker = -2
                tmp = curBladeX[:]  # copy
                tmp.reverse()

                # locate where the steady increase begins
                for j in range(k-1):
                    if tmp[j] - tmp[j+1] == 1:
                        marker = marker - 1
                    else:
                        break

                if marker < -k:
                    blades.append(tuple(curBladeX))
                    continue

                # replace
                blades.append(tuple(curBladeX))
                curBladeX[marker:] = list(range(
                    curBladeX[marker] + 1, curBladeX[marker] + 1 - marker))

    return blades


def Cl(p=0, q=0, sig=None, names=None, firstIdx=1, mvClass=MultiVector):
    """Returns a Layout and basis blades for the geometric algebra Cl_p,q.

    The notation Cl_p,q means that the algebra is p+q dimensional, with
    the first p vectors with positive signature and the final q vectors
    negative.

    Cl(p, q=0, names=None, firstIdx=0) --> Layout, {'name': basisElement, ...}
    """
    if sig is None:
        sig = [+1]*p + [-1]*q
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
    >>>randomMV(layout, min=-2.0, max=2.0, grades=None, uniform=None,n=2)

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

        gp(M,N) =  M * N

        M and N must be from the same layout

        This is useful in calculating series of products, with `reduce()`
        for example

        >>>Ms = [M1,M2,M3] # list of multivectors
        >>>reduce(gp, Ms) #  == M1*M2*M3

        """

        return M*N


def ip(M, N):
        """
        Inner product function

        ip(M,N) =  M | N

        M and N must be from the same layout

        """

        return M ^ N


def op(M, N):
        """
        Outer product function

        op(M,N) =  M ^ N

        M and N must be from the same layout

        This is useful in calculating series of products, with `reduce()`
        for example

        >>>Ms = [M1,M2,M3] # list of multivectors
        >>>reduce(op, Ms) #  == M1^M2^M3

        """

        return M ^ N


def conformalize(layout, added_sig=[1,-1]):
    '''
    Conformalize a Geometric Algebra

    Given the `Layout` for a GA of signature (p,q), this
    will produce a GA of signature (p+1,q+1), as well as
    return a new list of blades and some `stuff`. `stuff`
    is a dict containing the null basis blades, and some
    up/down functions for projecting in/out of the CGA.



    Parameters
    -------------
    layout: `clifford.Layout`
         layout of the GA to conformalize

    Returns
    ---------
    layout_c:  `clifford.Layout`
        layout of the conformalized GA
    blades_c: dict
        blades for the CGA
    stuff: dict
        dict containing the following:
            * ep - postive basis vector added
            * en - negative basis vector added
            * eo - zero vector of null basis (=.5*(en-ep))
            * einf - infinity vector of null basis (=en+ep)
            * E0 - minkowski bivector (=einf^eo)
            * I_ga - pseudoscalar for conformalized GA in new layout
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
    layout_c, blades_c = Cl(sig=sig_c)
    basis_vectors = layout_c.basis_vectors
    added_keys = sorted(layout_c.basis_vectors.keys())[-2:]
    ep, en = [basis_vectors[k] for k in added_keys]

    # setup  null basis, and minkowski subspace bivector
    eo = .5 ^ (en - ep)
    einf = en + ep
    E0 = einf ^ eo
    I_ga = layout_c.pseudoScalar*E0
    #  some  convenience functions
    def up(x):
        return x + (.5 ^ ((x**2)*einf)) + eo

    def homo(x):
        return x*(-x | einf).normalInv()  # homogenise conformal vector

    def down(x):
        return (homo(x) ^ E0)*E0

    stuff = {}
    stuff.update({
        'ep': ep, 'en': en, 'eo': eo, 'einf': einf, 'E0': E0,
        'up': up, 'down': down, 'homo': homo,'I_ga':I_ga})

    return layout_c, blades_c, stuff
