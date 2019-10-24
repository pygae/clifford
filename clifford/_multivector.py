import numbers
import math
from typing import List, Set, Tuple

import numpy as np

import clifford as cf
from . import (
    general_exp,
    grades_present,
    compute_reordering_sign_and_canonical_form,
)


class MultiVector(object):
    """An element of the algebra

    Parameters
    -------------
    layout: instance of :class:`clifford.Layout`
        the layout of the algebra

    value : sequence of length ``layout.gaDims``
        the coefficients of the base blades

    Notes
    ------
    The following operators are overloaded:

    * ``A * B`` : geometric product
    * ``A ^ B`` : outer product
    * ``A | B`` : inner product
    * ``A << B`` : left contraction
    * ``~M`` : reversion
    * ``M(N)`` : grade or subspace projection
    * ``M[N]`` : blade projection
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

    def __array__(self) -> 'cf.MVArray':
        # we are a scalar, and the only appropriate dtype is an object array
        return cf.MVArray([self])

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
        """Geometric product :math:`MN` """

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
        """Right-hand geometric product, :math:`NM`"""

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
        r""" Outer product, :math:`M \wedge N` """

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
        r"""Right-hand outer product, :math:`N \wedge M` """

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
        r""" Inner product, :math:`M \cdot N` """

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
        """Division, :math:`M N^{-1}`"""

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
        """Right-hand division, :math:`N M^{-1}`"""

        other, mv = self._checkOther(other)
        if isinstance(other, np.ndarray):
            obj = self.__array__()
            return other / obj

        return other * self.inv()

    def __pow__(self, other) -> 'MultiVector':
        """Exponentiation of a multivector by an integer, :math:`M^{n}` """

        if not isinstance(other, (int, float)):
            raise ValueError("exponent must be a Python int or float")

        if abs(round(other) - other) > cf._eps:
            raise ValueError("exponent must have no fractional part")

        other = int(round(other))

        if other == 0:
            return 1

        newMV = self._newMV(np.array(self.value))  # copy

        for i in range(1, other):
            newMV = newMV * self

        return newMV

    def __rpow__(self, other) -> 'MultiVector':
        """Exponentiation of a real by a multivector, :math:`r^{M}`"""

        # Let math.log() check that other is a Python number, not something
        # else.

        # pow(x, y) == exp(y * log(x))
        newMV = general_exp(math.log(other) * self)

        return newMV

    def __lshift__(self, other) -> 'MultiVector':
        """
        The ``<<`` operator is the left contraction
        """
        return self.lc(other)

    # unary

    def __neg__(self) -> 'MultiVector':
        """Negation, :math:`-M`"""

        newValue = -self.value

        return self._newMV(newValue)

    def as_array(self) -> np.ndarray:
        return self.value

    def __pos__(self) -> 'MultiVector':
        """Positive (just a copy), :math:`+M` """

        newValue = self.value + 0  # copy

        return self._newMV(newValue)

    def mag2(self) -> numbers.Number:
        """Magnitude (modulus) squared, :math:`{|M|}^2`

        Note in mixed signature spaces this may be negative
        """
        mv_val = self.layout.gmt_func(self.layout.adjoint_func(self.value), self.value)
        return mv_val[0]

    def __abs__(self) -> numbers.Number:
        """Magnitude (modulus), :math::`|M|`

        This is ``sqrt(abs(~M*M))``.

        The abs inside the sqrt is need for spaces of mixed signature
        """

        return np.sqrt(abs(self.mag2()))

    def adjoint(self) -> 'MultiVector':
        r"""Adjoint / reversion, :math:`\tilde M`

        Aliased as ``~M`` to reflect :math:`\tilde M`, one of several
        conflicting notations.

        Note that ``~(N * M) == ~M * ~N``.
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
            sign, blade = compute_reordering_sign_and_canonical_form(key, self.layout.sig,
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
            sign, blade = compute_reordering_sign_and_canonical_form(key, self.layout.sig,
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
            sign, blade = compute_reordering_sign_and_canonical_form(key, self.layout.sig,
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
        p = cf._print_precision

        for grade, name, coeff in zip(self.layout.gradeList, self.layout.names, self.value):
            # if we have nothing yet, don't use + and - as operators but
            # use - as an unary prefix if necessary
            if s:
                seps = (' + ', ' - ')
            else:
                seps = ('', '-')

            if abs(coeff) >= cf._eps:
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

        if cf._pretty:
            return self.__str__()

        s = "MultiVector(%s, value=%s)" % (
            repr(self.layout), list(self.value))
        return s

    def __bool__(self) -> bool:
        """Instance is nonzero iff at least one of the coefficients
        is nonzero.
        """

        nonzeroes = np.absolute(self.value) > cf._eps

        if nonzeroes.any():
            return True
        else:
            return False

    def __eq__(self, other) -> bool:
        other, mv = self._checkOther(other)
        if not mv:
            return NotImplemented

        if (np.absolute(self.value - other.value) < cf._eps).all():
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

        eps defaults to the current value of the global cf._eps.
        """

        if eps is None:
            eps = cf._eps

        mask = np.absolute(self.value) > eps

        # note element-wise multiplication
        self.value = mask * self.value

        return self

    def round(self, eps=None) -> 'MultiVector':
        """Rounds all coefficients according to Python's rounding rules.

        eps defaults to the current value of the global cf._eps.
        """

        if eps is None:
            eps = cf._eps

        self.value = np.around(self.value, eps)

        return self

    # Geometric Algebraic functions
    def lc(self, other) -> 'MultiVector':
        r"""The left-contraction of two multivectors, :math:`M\rfloor N`"""

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
            if abs(self.value[i]) < cf._eps:
                continue
            else:
                return False

        return True

    def isBlade(self) -> bool:
        """Returns true if multivector is a blade.
        """
        if len(self.grades()) != 1:
            return False

        return self.isVersor()

    def isVersor(self) -> bool:
        """Returns true if multivector is a versor.
        From Leo Dorsts GA for computer science section 21.5, definition from 7.6.4
        """
        Vhat = self.gradeInvol()
        Vrev = ~self
        Vinv = Vrev/(self*Vrev)[0]

        # Test if the versor inverse (~V)/(V * ~V) is truly the inverse of the
        # multivector V
        if grades_present(Vhat*Vinv, 0.000001) != {0}:
            return False
        if not np.sum(np.abs((Vhat*Vinv).value - (Vinv*Vhat).value)) < 0.0001:
            return False

        # applying a versor (and hence an invertible blade) to a vector should
        # not change the grade
        if not all(
            grades_present(Vhat*e*Vrev, 0.000001) == {1}
            for e in cf.basis_vectors(self.layout).values()
        ):
            return False

        return True

    def grades(self) -> Set[int]:
        """Return the grades contained in the multivector.

        .. versionchanged:: 1.1.0
            Now returns a set instead of a list
        """
        return grades_present(self, cf._eps)

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
        r"""Return the (mostly) normalized multivector.

        The _mostly_ comes from the fact that some multivectors have a
        negative squared-magnitude.  So, without introducing formally
        imaginary numbers, we can only fix the normalized multivector's
        magnitude to +-1.

        :math:`\frac{M}{|M|}` up to a sign
        """

        return self / np.sqrt(abs(self.mag2()))

    def leftLaInv(self) -> 'MultiVector':
        """Return left-inverse using a computational linear algebra method
        proposed by Christian Perwass.
        """
        return self._newMV(self.layout.inv_func(self.value))

    def normalInv(self) -> 'MultiVector':
        r"""The inverse of itself if :math:`M \tilde M = |M|^2`.

        ..math::

            M^{-1} = \tilde M / (M \tilde M)
        """

        Madjoint = ~self
        MadjointM = (Madjoint * self)

        if MadjointM.isScalar() and abs(MadjointM[()]) > cf._eps:
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
        r"""The dual of the multivector against the given subspace I, :math:`\tilde M = MI^{-1}`

        I defaults to the pseudoscalar.
        """
        if I is None:
            return self.layout.MultiVector(value=self.layout.dual_func(self.value))
        else:
            Iinv = I.inv()

        return self * Iinv

    def commutator(self, other) -> 'MultiVector':
        r"""The commutator product of two multivectors.

        :math:`[M, N] = M \cross N = (MN + NM)/2`
        """

        return ((self * other) - (other * self)) / 2

    x = commutator

    def anticommutator(self, other) -> 'MultiVector':
        """The anti-commutator product of two multivectors, :math:`(MN + NM)/2` """

        return ((self * other) + (other * self)) / 2

    def gradeInvol(self) -> 'MultiVector':
        r"""The grade involution of the multivector.

        .. math::
            M^* = \sum_{i=0}^{\text{dims}}
                  {(-1)^i \left<M\right>_i}
        """

        signs = np.power(-1, self.layout.gradeList)

        newValue = signs * self.value

        return self._newMV(newValue)

    @property
    def even(self) -> 'MultiVector':
        '''
        Even part of this multivector

        defined as
        ``M + M.gradInvol()``
        '''
        return .5*(self + self.gradeInvol())

    @property
    def odd(self) -> 'MultiVector':
        '''
        Odd part of this mulitvector

        defined as
        ``M +- M.gradInvol()``
        '''
        return .5*(self - self.gradeInvol())

    def conjugate(self) -> 'MultiVector':
        """The Clifford conjugate (reversion and grade involution).

        :math:`M^*` = ``(~M).gradeInvol()``
        """
        return (~self).gradeInvol()

    # Subspace operations
    def project(self, other) -> 'MultiVector':
        r"""Projects the multivector onto the subspace represented by this blade.

        :math:`P_A(M) = (M \rfloor A) A^{-1}`
        """

        other, mv = self._checkOther(other, coerce=True)

        if not self.isBlade():
            raise ValueError("self is not a blade")

        return other.lc(self) * self.inv()

    def factorise(self) -> Tuple[List['MultiVector'], numbers.Number]:
        """
        Factorises a blade into basis vectors and an overall scale.

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

        # only one grade, since this is a blade
        gr, = self.grades()

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
                if len(thisBasis) == gr:  # we have a complete set
                    break

        return thisBasis

    def join(self, other) -> 'MultiVector':
        r"""The join of two blades.

        :math:`J = A \wedge B`
        """

        other, mv = self._checkOther(other)

        grSelf = self.grades()
        grOther = other.grades()

        if len(grSelf) == len(grOther) == 1:
            # both blades
            grSelf, = grSelf
            grOther, = grOther

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

            if grSelf >= grOther:
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
        r"""The meet of two blades.

        Computation is done with respect to a subspace that defaults to
        the join if none is given.

        :math:`M \vee_i N = M_i^{-1}N`
        """

        other, mv = self._checkOther(other)

        r = self.grades()
        s = other.grades()

        if len(r) > 1 or len(s) > 1:
            raise ValueError("not blades")

        if subspace is None:
            subspace = self.join(other)

        return (self * subspace.inv()) | other

    def astype(self, *args, **kwargs):
        """
        Change the underlying scalar type of this vector.

        Can be used to force lower-precision floats or integers

        See `np.ndarray.astype` for argument descriptions.
        """
        return self._newMV(self.value.astype(*args, **kwargs))
