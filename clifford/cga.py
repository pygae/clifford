'''
.. currentmodule:: clifford.cga

========================================
cga (:mod:`clifford.cga`)
========================================

Object Oriented Conformal Geometric Algebra.

Examples
-----------
>>> from clifford import Cl
>>> from clifford.cga import CGA
>>> g3, blades = Cl(3)
>>> locals().update(blades)
>>> g3c = CGA(g3)
>>> C = g3c.round(3)             # create random sphere
>>> T = g3c.translation(e1+e2)   # create translation
>>> C_ = T(C)                    # translate the sphere
>>> C_.center                    # compute center of sphere
-(1.0^e4) - (1.0^e5)

The CGA
========

.. autosummary::
    :toctree: generated/

    CGA


Objects
================

.. autosummary::
    :toctree: generated/

    Flat
    Round

Operators
================

.. autosummary::
    :toctree: generated/

    Rotation
    Dilation
    Translation
    Transversion

Meta-Class
===========

.. autosummary::
    :toctree: generated/

    CGAThing

'''

from functools import reduce
from typing import overload
from . import conformalize, op, gp, MultiVector, Cl
from numpy import zeros, e, log
from numpy.random import rand
import math

try:
    import pyganja as ganja
    pyganja_available = True
except ImportError:
    pyganja_available = False


class CGAThing(object):
    '''
    base class for cga objects and operators.

    maps versor product to `__call__`.
    '''
    def __init__(self, cga: 'CGA') -> None:
        self.cga = cga
        self.layout = cga.layout

    @overload
    def __call__(self, other: MultiVector) -> MultiVector: pass  # noqa: F811

    @overload
    def __call__(self, other: 'CGAThing') -> 'CGAThing': pass  # noqa: F811

    def __call__(self, other):  # noqa: F811
        if isinstance(other, MultiVector):
            if other.grades() == {1}:
                null = self.cga.null_vector(other)
                return self.mv*null*~self.mv
            return self.mv*other*~self.mv
        else:
            klass = other.__class__
            return klass(self.cga, self.mv*other.mv*~self.mv)

    def inverted(self) -> MultiVector:
        '''
        inverted version of this thing.

        self -> ep*self*ep

        where ep is the positive added basis vector
        '''
        return self.cga.ep * self.mv * self.cga.ep

    def involuted(self) -> MultiVector:
        '''
        inverted version of this thing.

        self -> E0*self*E0

        where E0 is the added minkowski bivector
        '''
        return self.cga.E0 * self.mv * self.cga.E0


# Objects
class Flat(CGAThing):
    '''
        A line, plane, or hyperplane.

        Typically constructed as method of existing cga, like `cga.flat()`

        multivector is accessable by `mv` property

        Parameters
        -----------
        cga : `CGA`
            the cga object

        args : [int, Multivector, Multivectors]
            * if nothing supplied, generate a flat of highest dimension
            * int: dimension of flat  (2=line, 3=plane, etc)
            * Multivector : can be
              * existing Multivector representing the Flat
              * vectors on the flat

        Examples
        ----------

        >>> cga = CGA(3)
        >>> locals().update(cga.blades)
        >>> F = cga.flat()               # from None
        >>> F = cga.flat(2)              # from dim of space
        >>> F = cga.flat(e1, e2)         # from points
        >>> F = cga.flat(cga.flat().mv)  # from existing multivector
        '''
    # could inherent some generic CGAObject class
    def __init__(self, cga, *args) -> None:
        super().__init__(cga)
        self.einf = self.cga.einf  # we use this alot

        if len(args) == 0:
            # generate random highest dimension flat
            nulls = [self.cga.null_vector() for k in range(self.layout.dims-2)]
            self.mv = reduce(op, nulls + [self.einf])

        elif len(args) == 1:
            # from existing multivector
            if isinstance(args[0], MultiVector):
                self.mv = args[0]

            # generate random flat  for given  dimension
            elif isinstance(args[0], int):
                dim = args[0]
                points = [self.cga.base_vector() for k in range(dim+1)]
                points = list(map(self.cga.up, points))
                self.mv = reduce(op, points + [self.einf])

        # from vectors on flat
        else:
            nulls = map(self.cga.null_vector, args)
            if self.einf not in nulls:
                nulls = list(nulls)+[self.einf]

            self.mv = reduce(op, nulls)

        self.mv = self.mv.normal()

    def __repr__(self) -> str:
        return '%i-Flat' % self.dim


class Round(CGAThing):
    '''
    A point pair, circle, sphere or hyper-sphere.

    Typically constructed as method of existing cga, like `cga.round()`

    multivector is accessable by `mv` property

    Parameters
    -----------
    cga : `CGA`
        the cga object

    args : [int, Multivector, Multivectors]
        * if nothing supplied, generate a round of highest dimension
        * int: dimension of flat  (2=point pair, 3=circle, etc)
        * Multivector : can be
          * existing Multivector representing the round
          * vectors on the round

    Examples
    ----------

    >>> cga = CGA(3)
    >>> locals().update(cga.blades)
    >>> cga.round()               # from None
    Sphere
    >>> cga.round(2)              # from dim of space
    Sphere
    >>> cga.round(e1, e2, -e1)    # from points
    Circle
    >>> cga.round(cga.flat().mv)  # from existing multivector
    Sphere
    '''
    # could inherent some generic CGAObject class
    def __init__(self, cga, *args) -> None:
        super().__init__(cga)

        if len(args) == 0:
            # generate random highest dimension round
            nulls = [self.cga.null_vector() for k in range(self.layout.dims-1)]
            self.mv = reduce(op, nulls)

        elif len(args) == 1:
            # from existing multivector
            if isinstance(args[0], MultiVector):
                self.mv = args[0]

            # generate random round  for given  dimension
            elif isinstance(args[0], int):
                dim = args[0]
                points = [self.cga.base_vector() for k in range(dim+2)]
                points = map(self.cga.up, points)
                self.mv = reduce(op, points)

            # from center, radius tuple
            else:
                if len(args[0]) == 2:
                    center, radius = args[0]
                    center = self.cga.null_vector(center)
                    dual_round = (center - .5*radius**2*self.cga.einf)
                    self.mv = dual_round.normal().dual()

        # from vectors on round
        else:
            nulls = map(self.cga.null_vector, args)
            self.mv = reduce(op, nulls)

        self.mv = self.mv.normal()

    def from_center_radius(self, center, radius):
        '''
        construct a round from center/radius
        '''
        center = self.cga.null_vector(center)
        self.mv = (center - .5*radius**2*self.cga.einf).normal().dual()
        return self

    def __repr__(self) -> str:
        names = {4: 'Sphere', 3: 'Circle', 2: 'Point Pair', 1: 'Point'}
        if self.dim <= 4:
            return names[self.dim + 2]
        else:
            return '%i-Round' % (self.dim + 2)

    @property
    def dim(self):
        '''
        dimension of this round
        '''
        gr, = self.mv.grades()
        return gr - 2

    @property
    def center(self):
        '''
        center of this round, as a null vector
        '''
        return self.mv * self.cga.einf * self.mv

    @property
    def center_down(self):
        '''
        center of this round, as a down-projected  vector (in I_base)

        (but still in cga's layout)
        '''
        return self.cga.down(self.center)

    @property
    def radius(self):
        '''
        radius of the round (a float)
        '''
        dual_sphere = self.dual
        dual_sphere /= (-dual_sphere | self.cga.einf)
        return math.sqrt(abs(dual_sphere * dual_sphere))

    @property
    def dual(self):
        '''
        self.mv* self.layout.I
        '''
        return self.mv * self.layout.I


# Operators
class Translation(CGAThing):
    '''
    A Translation

    Can be constructed from a vector in base space or a null
    vector, or nothing.

    Parameters
    ----------
    args : [none, `clifford.Multivector`]
        if none, a random translation will be generated
        several types of Multivectors can be used:

         * base vector - vector in base space
         * null vector
         * existing translation rotor

    Examples
    ----------
    >>> cga = CGA(3)
    >>> locals().update(cga.blades)
    >>> T = cga.translation()       # from None
    >>> T = cga.translation(e1+e2)  # from base vector
    >>> T = cga.translation(cga.up(e1+e2)) # from null vector
    >>> T = cga.translation(T.mv)  # from existing translation rotor
    '''
    def __init__(self, cga, *args) -> None:
        super().__init__(cga)

        if len(args) == 0:
            # generate generator!
            mv = 1 - self.cga.base_vector()*self.cga.einf/2.

        elif len(args) == 1:
            arg = args[0]
            if isinstance(arg, MultiVector):
                if arg.grades() == {1}:
                    # we have vector
                    mv = 1 - self.cga.straight_up(arg)*self.cga.einf/2.
                if arg.grades() == {0, 2}:
                    # we have ro tor
                    # TODO ensure its a translation
                    mv = args[0]
        else:
            raise ValueError('bad input')

        self.mv = mv

    def __repr__(self) -> str:
        return 'Translation'


class Dilation(CGAThing):
    '''
    A global dilation

    Parameters
    ----------
    args : [none, number]
        if none, a random dilation will be generated
        if a number, dilation of given amount

    Examples
    ----------
    >>> cga = CGA(3)
    >>> D = cga.dilation()          # from  none
    >>> D = cga.dilation(.4)        # from number
    '''
    def __init__(self, cga, *args) -> None:
        super().__init__(cga)

        if len(args) == 0:
            # generate a dilation
            args = [rand()]

        if len(args) == 1:
            arg = args[0]
            if isinstance(arg, MultiVector):
                if arg.grades() == {0, 2}:
                    # we have a rotor
                    mv = arg
                if arg.grades == {0}:
                    arg = float(arg)

            if arg < 0:
                raise(ValueError('dilation should be positive'))

            mv = e**((-log(arg)/2.)*(self.cga.E0))

        else:
            raise ValueError('bad input')

        self.mv = mv

    def __repr__(self) -> str:
        return 'Dilation'


class Rotation(CGAThing):
    '''
    A Rotation

    Can be constructed from a generator, rotor, or none

    Parameters
    ----------
    args : [none, `clifford.Multivector`]
        if none, a random translation will be generated
        several types of Multivectors can be used:

         * bivector   - interpreted as the generator
         * existing translation rotor

    Examples
    ----------
    >>> cga = CGA(3)
    >>> locals().update(cga.blades)
    >>> R = cga.rotation()          # from None
    >>> R = cga.rotation(e12+e23)   # from bivector
    >>> R = cga.rotation(R.mv)   # from bivector
    '''
    def __init__(self, cga, *args) -> None:
        super().__init__(cga)

        if len(args) == 0:
            # generate a rotation

            U = self.layout.randomMV()(2)
            U = self.cga.I_base.project(U)
            self.mv = e**(U)

        elif len(args) == 1:
            arg = args[0]
            if isinstance(arg, MultiVector):
                if arg.grades() == {0, 2}:
                    # we have a rotor
                    self.mv = arg

                elif arg.grades() == {2}:
                    # we have a bivector, make sure its in base space
                    if arg^self.cga.I_base != 0:
                        arg = self.cga.I_base.project(arg)
                    self.mv = e**(arg)
                else:
                    # multivector has improper grade
                    raise ValueError('bad input')
            else:
                # arg isnt a multivector
                raise ValueError('bad input')

        else:
            # more than 1 arg
            raise ValueError('bad input')

    def __repr__(self) -> str:
        return 'Rotation'


class Transversion(Translation):
    '''
    A Transversion

    A transversion is a combination of an inversion-translation-inversion,
    or in other words an inverted translation operator. This inherits
    from `Translation`

    Can be constructed from a vector in base space or a null
    vector, or nothing.

    Parameters
    ----------
    args : [none, `clifford.Multivector`]
        if none, a random transversion will be generated
        several types of Multivectors can be used:

         * base vector - vector in base space
         * null vector
         * existing transversion rotor

    Examples
    ----------
    >>> cga = CGA(3)
    >>> locals().update(cga.blades)
    >>> K = cga.transversion()       # from None
    >>> K = cga.transversion(e1+e2)  # from base vector
    >>> K = cga.transversion(cga.up(e1+e2)) # from null vector
    >>> T = cga.translation()
    >>> K = cga.transversion(T.mv)  # from existing translation rotor
    '''
    def __init__(self, cga, *args) -> None:
        CGAThing.__init__(self, cga)
        self.mv = Translation(cga, *args).inverted()

    def __repr__(self) -> str:
        return 'Transversion'


class CGA(object):
    '''
    Conformal Geometric Algebra

    conformalizes the layout_orig,  and provides several
    methods and for objects/operators

    Parameters
    -----------
    layout_orig: [`clifford.Layout`, int]
        a layout for the *base* geometric algebra which is conformalized
        if given as an int, then generates a euclidean space of given
        dimension

    Examples
    ----------
    >>> from clifford import Cl
    >>> from clifford.cga import CGA
    >>> g3, blades  = Cl(3)
    >>> g3c = CGA(g3)
    >>> g3c = CGA(3)

    '''
    def __init__(self, layout_orig) -> None:
        if isinstance(layout_orig, int):
            layout_orig, blades = Cl(layout_orig)
        self.layout_orig = layout_orig
        self.layout, self.blades, stuff = conformalize(layout_orig)
        self.__dict__.update(stuff)

    # Objects
    def base_vector(self) -> MultiVector:
        '''
        random vector in the lower(original) space
        '''
        return self.I_base.project(self.layout.randomV())

    def null_vector(self, x=None) -> MultiVector:
        '''
        generates random null vector if x is None, or
        returns a null vector from base vector x, if  x^self.I_base ==0
        returns x,

        a null vector will lay on the horisphere
        '''
        if x is None:
            return self.up(self.base_vector())
        else:
            if x^self.I_base == 0:
                return self.up(x)
            return x

    def round(self, *args) -> Round:
        '''
        see :class:`Round`
        '''
        return Round(self, *args)

    def flat(self, *args) -> Flat:
        '''
        see :class:`Flat`
        '''
        return Flat(self, *args)

    #  Operators
    def translation(self, *args) -> Translation:
        '''
        see :class:`Translation`
        '''
        return Translation(self, *args)

    def transversion(self, *args) -> Transversion:
        '''
        see :class:`Transversion`
        '''
        return Transversion(self, *args)

    def dilation(self, *args) -> Dilation:
        '''
        see :class:`Dilation`
        '''
        return Dilation(self, *args)

    def rotation(self, *args) -> Rotation:
        '''
        see :class:`Rotation`
        '''
        return Rotation(self, *args)

    #  methods
    def straight_up(self, x) -> MultiVector:
        '''
        place a vector from layout_orig into this CGA, without up()
        '''
        return self.I_base.project(self.up(x))
