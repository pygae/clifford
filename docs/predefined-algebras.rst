Predefined Algebras
===================

The easiest way to get started with ``clifford`` is to use one of several predefined algebras:

* .. module:: clifford.g2

  ``g2``: 2D Euclidean, ``Cl(2)``. See :doc:`tutorials/g2-quick-start` for some examples.
* .. module:: clifford.g3

  ``g3``: 3D Euclidean, ``Cl(3)``. See :doc:`tutorials/g3-algebra-of-space` for some examples.
* .. module:: clifford.g4

  ``g4``: 4D Euclidean, ``Cl(4)``.
* .. module:: clifford.g2c

  ``g2c``: Conformal space for G2, ``Cl(3, 1)``. See :doc:`tutorials/cga/index` for some examples.
* .. module:: clifford.g3c

  ``g3c``: Conformal space for G3, ``Cl(4, 1)``.
* .. module:: clifford.pga

  ``pga``: Projective space for G3 ``Cl(3, 0, 1)``.
* .. module:: clifford.gac

  ``gac``: Geometric Algebra for Conics, ``Cl(5, 3)``.
* .. module:: clifford.dpga

  ``dpga``: Double PGA also referred to as the Mother Algebra, ``Cl(4, 4)``.
* .. module:: clifford.dg3c

  ``dg3c``: Double Conformal Geometric Algebra, effectively two g3c algebras glued together ``Cl(8, 2)``.



.. currentmodule:: clifford.<predefined>

By using the pre-defined algebras in place of calling ``Cl`` directly, you will often find that your program starts up faster.

.. data:: clifford.<predefined>.e<ijk>

    All of these modules expose the basis blades as attributes, and can be used like so

    .. ipython::

        In [138]: from clifford import g2

        @doctest
        In [138]: g2.e1 * g2.e2
        Out[138]: (1^e12)

Additionally, they define the following attributes, which contain the return values of :func:`clifford.Cl`:

.. data:: layout

    The associated :class:`clifford.Layout`

    .. ipython::

        @doctest
        In [138]: g2.layout
        Out[138]: Layout([1, 1],
               ids=BasisVectorIds.ordered_integers(2),
               order=BasisBladeOrder.shortlex(2),
               names=['', 'e1', 'e2', 'e12'])

.. data:: blades

    A shorthand for :meth:`Layout.blades`

    .. ipython::

        @doctest
        In [138]: g2.blades
        Out[138]: {'': 1, 'e1': (1^e1), 'e2': (1^e2), 'e12': (1^e12)}

For interactive use, it's very handy to use ``import *``

.. ipython::

    In [138]: from clifford.g2 import *

    @doctest
    In [138]: e1, e2, e12
    Out[138]: ((1^e1), (1^e2), (1^e12))

For the conformal layouts :mod:`~clifford.g2c` and :mod:`~clifford.g3c`, the full contents of the ``stuff`` result of :func:`clifford.conformalize` is also exposed as members of the module.
