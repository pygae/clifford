"""
Tools for interpreting conformal blades

.. currentmodule:: clifford.tools.classify

.. autofunction:: classify

The return type is a :class:`Blade`:

.. autoclass:: Blade
    :members:

The subclasses below are the four categories to which all blades belong, where
:math:`E` is a euclidean blade, and :math:`T_p[X]` represents a translation of
the conformal blade :math:`X` by the euclidean vector :math:`p`.

.. autoclass:: Direction

.. autoclass:: Flat

.. autoclass:: DualFlat

.. autoclass:: Round

These can be constructed directly, and will attempt to show a grade-specific
interpretation::

    >>> from clifford.g3c import *
    >>> Round(location=e1, direction=e1^e2, radius=1)
    Circle(direction=(1^e12), location=(1^e1), radius=1)
    >>> Round(direction=e1^e2^e3, location=e1, radius=1)
    Sphere(direction=(1^e123), location=(1^e1), radius=1)


Aliased types
-------------

In addition, aliases are created for specific grades of the above types, with
more convenient names:

.. autoclass:: Tangent
    :show-inheritance:

.. autoclass:: Point
    :show-inheritance:

.. autoclass:: PointFlat
    :show-inheritance:

.. autoclass:: Line
    :show-inheritance:

.. autoclass:: Plane
    :show-inheritance:

.. autoclass:: PointPair
    :show-inheritance:

.. autoclass:: Circle
    :show-inheritance:

.. autoclass:: Sphere
    :show-inheritance:

.. autoclass:: InfinitePoint
    :show-inheritance:



"""
import math
import sys
from typing import Optional
import copyreg

from clifford._multivector import MultiVector
from clifford._conformal_layout import ConformalLayout

__all__ = [
    'classify',

    'Blade',
    'Direction',
    'Flat',
    'DualFlat',
    'Round',
    'Tangent',
    'Point',
    'PointFlat',
    'Line',
    'Plane',
    'PointPair',
    'Circle',
    'Sphere',
    'InfinitePoint',
]


def _only_grade(mv: MultiVector, when_zero: int = 0) -> int:
    gs = mv.grades()
    if len(gs) > 1:
        raise ValueError("MultiVector has more than one grade")
    elif not gs:
        return when_zero
    else:
        g, = gs
        return g


class _GradedTypesMeta(type):
    """
    Metaclass for putting the grade number in the class name.

    Parameters
    ----------
    override : bool
        Used when creating an override of a particular grade, such as
        ``class Line(Flat[3], override=True)``, which will make ``Flat[3]``
        return ``Line`` in future. Each grade can be overriden exactly once,
        and should not be outside of this file.
    """
    def __new__(metacls, names, bases, dict, override=False):
        cls = type.__new__(metacls, names, bases, dict)
        if not cls.__instantiated:
            # for `class Flat` below
            cls.__graded_types = {}
            cls.__overriden = set()
        elif cls._grade not in cls.__graded_types:
            # for `Flat[n]`
            pass
        elif override:
            # for `class Line(Flat[3], override=True)`
            if cls._grade not in cls.__overriden:
                cls.__overriden.add(cls._grade)
                cls.__graded_types[cls._grade] = cls
            else:
                raise ValueError("Cannot override this grade multiple times")
        return cls

    if sys.version_info < (3, 6):
        # implied in newer versions of python
        def __init__(metacls, names, bases, dict, override=False):
            super().__init__(names, bases, dict)

    @property
    def __instantiated(cls) -> bool:
        """ true if this class is of the form `Flat[3]`, false if it is `Flat` """
        return hasattr(cls, '_grade')

    def __getitem__(cls, grade) -> '_GradedTypesMeta':
        """ Get the grade-specific type. Used to implement ``Blade[n]``. """
        if cls.__instantiated:
            raise TypeError("Cannot index multiple times")
        try:
            return cls.__graded_types[grade]
        except KeyError:
            # augment the bases with the instantiated versions of themselves
            bases = (cls,)
            for b in cls.__bases__:
                if isinstance(b, _GradedTypesMeta) and not b.__instantiated:
                    bases = (b[grade],) + bases
            subcls = cls.__graded_types[grade] = type(
                '{}[{}]'.format(cls.__name__, grade), bases, dict(_grade=grade)
            )
            return subcls

    def _expected_grade(cls, *args, **kwargs) -> Optional[int]:
        """ Determine the expected grade from the argument values

        This is used to make ``Direction(e123)`` give ``Direction[3](e123)``,
        and to make ``Direction[1](e123)`` an error.

        Override in subclasses. The default is to not allow inference, and not
        perform grade checks, which is indicated by returning None.
        """
        return None

    def __call__(cls, *args, **kwargs):
        """ Overload to prevent `Flat(...)` being used """
        g = cls._expected_grade(*args, **kwargs)
        if not cls.__instantiated:
            if g is None:
                raise TypeError("Must construct as {}[n](...)".format(cls.__name__))
            return cls[g].__call__(*args, **kwargs)
        if g is not None and g != cls._grade:
            raise ValueError(
                "Grade inferred from arguments ({}) does not match class grade "
                "({})"
                .format(g, cls._grade)
            )
        return super().__call__(*args, **kwargs)

    # __reduce__ doesn't work in a metaclass, we have to use copyreg instead
    def _pickle_override(cls):
        last_base = cls.__bases__[-1]
        # if the last base class is uninstantiated, we deduce it came from
        # __getitem__ above
        if cls.__instantiated and not last_base.__instantiated:
            # tell pickle to use our square bracket syntax to recover the class
            return (_GradedTypesMeta.__getitem__, (last_base, cls._grade))
        else:
            # regular pickling
            return cls.__qualname__


# make sure we have a chance to take part in pickling
copyreg.pickle(_GradedTypesMeta, _GradedTypesMeta._pickle_override)


class Blade(metaclass=_GradedTypesMeta):
    """
    Base class for providing interpretation of blades.

    Note that thanks to the unual metaclass, this class and its subclasses are
    have grade-specific specializations, eg ``Blade[2]`` is a type for blades of
    grade 2.

    Attributes
    ----------
    layout : ~clifford.Layout
        The layout to which this blade belongs
    """
    def __init__(self, layout):
        self.layout = layout

    def _repr_skip_members(self):
        return {'layout'}

    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)

    def __repr__(self):
        members = self.__dict__.copy()
        for name in self._repr_skip_members():
            members.pop(name)
        return "{}({})".format(type(self).__name__, ", ".join(
            "{}={!r}".format(k, v) for k, v in members.items()
        ))

    def _repr_pretty_(self, p, cycle):
        members = self.__dict__.copy()
        for name in self._repr_skip_members():
            members.pop(name)

        prefix = '{}('.format(type(self).__name__)
        with p.group(len(prefix), prefix, ')'):
            is_first = True
            for k, v in members.items():
                if not is_first:
                    p.text(',')
                    p.breakable()
                p.text("{}={!r}".format(k, v))
                is_first = False

    def _translate(self, t, x):
        """ Internal helper to translate x (conformal) by t (euclidean vector) """
        einf = self.layout.einf
        versor = 1 - t*einf/2
        return versor * x * ~versor

    @property
    def mv(self) -> MultiVector:
        """ Convert this back into its GA representation """
        raise NotImplementedError   # pragma: no cover


class Direction(Blade):
    r""" :math:`En_\infty`

    Attributes
    ----------
    direction : ~clifford.MultiVector
        The euclidean direction, :math:`E`
    """
    @classmethod
    def _expected_grade(cls, direction):

        return _only_grade(direction, when_zero=-1) + 1

    def __init__(self, direction):
        super().__init__(direction.layout)
        self.direction = direction

    @property
    def mv(self):
        return self.direction^self.layout.einf


class Flat(Blade):
    r""" :math:`T_p[n_o \wedge (En_\infty)]`

    Attributes
    ----------
    direction : ~clifford.MultiVector
        The euclidean direction, :math:`E`
    location : ~clifford.MultiVector
        The closest point on this flat to the origin, :math:`p`, as a euclidean
        vector.
    """
    @classmethod
    def _expected_grade(cls, direction, *args, **kwargs):
        return _only_grade(direction, when_zero=-2) + 2

    def __init__(self, direction, location):
        super().__init__(direction.layout)
        self.direction = direction
        self.location = location

    @property
    def mv(self):
        return self._translate(self.location, self.layout.eo^self.direction^self.layout.einf)


class DualFlat(Blade):
    """ Dual of :class:`Flat`

    Attributes
    ----------
    flat : Flat
        The flat this is the dual of
    """
    @classmethod
    def _expected_grade(cls, flat):
        return flat.layout.dims - type(flat)._grade

    def __init__(self, flat):
        super().__init__(flat.layout)
        self.flat = flat

    @property
    def mv(self):
        return self.flat.mv.dual()


class Round(Blade):
    r""" :math:`T_p[(n_o + \frac{1}{2}\rho^2 n_\infty)E]`

    Attributes
    ----------
    direction : ~clifford.MultiVector
        The euclidean direction, :math:`E`
    location : ~clifford.MultiVector
        The euclidean center, :math:`p`
    radius : float or complex
        The radius, :math:`\rho`, which may be imaginary
    """
    @classmethod
    def _expected_grade(cls, direction, *args, **kwargs):
        return _only_grade(direction) + 1

    def __init__(self, direction, location, radius):
        super().__init__(direction.layout)
        self.direction = direction
        self.location = location
        self.radius = radius

    @property
    def mv(self):
        return self._translate(self.location, (self.layout.eo + (self.radius*self.radius/2).real*self.layout.einf)^self.direction)


# specializations
class Tangent(Round):
    r""" A :class:`Round` of radius 0, :math:`T_p[n_o E]` """
    def __init__(self, direction, location):
        super().__init__(direction, location, 0)

    def _repr_skip_members(self):
        return super()._repr_skip_members() | {'radius'}


class Point(Tangent[1], override=True):
    r""" A conformal point, :math:`A` """


class PointFlat(Flat[2], override=True):
    r""" A flat point, :math:`A \wedge n_\infty` """


class Line(Flat[3], override=True):
    r""" A line, :math:`A \wedge B \wedge n_\infty` """


class Plane(Flat[4], override=True):
    r""" A line, :math:`A \wedge B \wedge C \wedge n_\infty` """


class PointPair(Round[2], override=True):
    r""" A point pair, :math:`A \wedge B` """


class Circle(Round[3], override=True):
    r""" A circle, :math:`A \wedge B \wedge C` """


class Sphere(Round[4], override=True):
    r""" A sphere, :math:`A \wedge B \wedge C \wedge D` """


class InfinitePoint(Direction[1], override=True):
    r""" A scalar multiple of :math:`n_\infty` """


def _sqrt(x):
    """ like cmath.sqrt, but leave real numbers as float """
    if x < 0:
        return math.sqrt(-x)*1j
    else:
        return math.sqrt(x)


def classify(x) -> Blade:
    """
    Classify a conformal multivector into a parameterized geometric description.

    The multivector should be from a :class:`~clifford.ConformalLayout`, such
    as the one returned by :func:`clifford.conformalize`.

    Implemented based on the approach described in table 14.1 of
    *Geometric Algebra for Computer Science (Revised Edition)*.

    Example usage::

        >>> from clifford.g3c import *

        >>> classify(e1)
        DualFlat[1](flat=Plane(direction=-(1.0^e23), location=0))

        >>> classify(einf)
        InfinitePoint(direction=1.0)

        >>> classify(up(e1))
        Point(direction=1.0, location=(1.0^e1))

        >>> classify(up(3*e1)^up(4*e2))
        PointPair(direction=-(3.0^e1) + (4.0^e2), location=(1.5^e1) + (2.0^e2), radius=2.5)

        >>> classify(up(e1)^up(e2)^up(e1+2*e2))
        Circle(direction=-(2.0^e12), location=(1.0^e1) + (1.0^e2), radius=1.0)

        >>> classify(up(e1)^up(e2)^up(e1+2*e2)^einf)
        Plane(direction=-(2.0^e12), location=0)

        >>> classify(up(e1)^e2)
        Tangent[2](direction=(1.0^e2), location=(1.0^e1))

        # how the inheritance works
        >>> Point.mro()  # doctest: +SKIP
        [Point, Tangent[1], Round[1], Blade[1], Tangent, Round, Blade, object]

    The reverse of this operation is :attr:`Blade.mv`.
    """
    if not isinstance(x.layout, ConformalLayout):
        raise ValueError("vector does not belong to a conformal algebra")

    einf = x.layout.einf
    eo = x.layout.eo
    down = x.layout.down

    try:
        g = _only_grade(x)
    except ValueError:
        raise ValueError("Cannot classify a combination of multiple grades")

    y = -einf | x
    if y == 0:
        if einf ^ x == 0:
            direction = x | -eo
            return Direction[g](direction=direction)
        else:
            x_orig = x * x.layout.I  # undual
            direction = (-einf | x_orig) | -eo
            flat = Flat[x.layout.dims - g](direction=direction, location=down((eo ^ x) * x.normalInv()))
            return DualFlat[g](flat=flat)
    else:
        if einf ^ x == 0:
            direction = y | -eo
            return Flat[g](direction=direction, location=down((eo | x) * x.normalInv()))
        else:
            direction = y^einf
            location = x * y.normalInv()
            rad2 = x * x.gradeInvol()
            direction = direction | -eo
            if rad2 == 0:
                return Tangent[g](direction=direction, location=down(location))
            else:
                return Round[g](direction=direction, location=down(location), radius=_sqrt(float(rad2)/float(y*y)))
