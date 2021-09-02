r"""
.. currentmodule:: clifford.numba

===============================================
numba extension support (:mod:`clifford.numba`)
===============================================

.. versionadded:: 1.4.0

This module provides :mod:`numba` extension types :class:`MultiVectorType` and
:class:`LayoutType` corresponding to :class:`~clifford.MultiVector` and
:class:`~clifford.Layout`.

You do not need to import this module to take advantage of these types; they
are needed only directly when writing numba overloads via
:func:`numba.extending.overload` and similar.

As a simple example, the following code defines a vectorized ``up()`` function
for :doc:`CGA </tutorials/cga/index>` ::

    from clifford.g3c import *

    @numba.njit
    def jit_up(x):
        return eo + x + 0.5*abs(x)**2*einf

    assert up(e1) == jit_up(e1)

Note that a rough equivalent to this particular function is provided elsewhere
as :func:`clifford.tools.g3c.fast_up`.

.. currentmodule:: clifford

Supported operations
--------------------
The following list of operations are supported in a jitted context:

* :class:`MultiVector`: A limited version of the constructor supporting only
  ``MultiVector(layout, value)`` and ``MultiVector(layout, dtype=dtype)``.
* :meth:`layout.MultiVector`, with the same caveats as above.
* :attr:`layout.dims`
* :attr:`layout.gaDims`
* :attr:`layout.sig`
* :attr:`MultiVector.value`
* :attr:`MultiVector.layout`
* Arithmetic:

  * :meth:`MultiVector.__add__`
  * :meth:`MultiVector.__sub__`
  * :meth:`MultiVector.__mul__`
  * :meth:`MultiVector.__xor__`
  * :meth:`MultiVector.__or__`
  * :meth:`MultiVector.__pow__`
  * :meth:`MultiVector.__truediv__`
  * :meth:`MultiVector.__invert__`
  * :meth:`MultiVector.__pos__`
  * :meth:`MultiVector.__neg__`

* :meth:`MultiVector.__call__`
* :meth:`MultiVector.mag2`
* :meth:`MultiVector.__abs__`
* :meth:`MultiVector.normal`
* :meth:`MultiVector.leftLaInv`
* :meth:`MultiVector.shirokov_inverse`
* :meth:`MultiVector.hitzer_inverse`
* :meth:`MultiVector.gradeInvol`
* :meth:`MultiVector.conjugate`
* :meth:`MultiVector.commutator`
* :meth:`MultiVector.anticommutator`
* :attr:`MultiVector.even`
* :attr:`MultiVector.odd`

Performance considerations
--------------------------
While the resulted jitted code is typically faster, there are two main
performance issues to consider. The first is the startup time of ``@jit``\ ing.
This can be quite substantial, although can be somewhat alleviated by using
the ``cache=True`` argument to :func:`numba.jit`.
The second is the time taken for numba to find the appropriate dispatch loop
given the Python types of its arguments, which adds overhead to every call.

``clifford`` tries as hard as possible to reduce this second overhead, by
using the undocumented ``_numba_type_`` attribute and keeping our own optimized
cache instead of going through the recommended
``@numba.extending.typeof_impl.register(LayoutType)`` mechanism.
However, this overhead can still be slow compared to elementary operations.
The following code is significantly impacted by this overhead::

    from clifford.g3c import *
    import numba

    @numba.njit
    def mul(a, b):
        return a * b

    # 286 ms, ignoring jitting time
    x = e1
    for i in range(10000):
        x = mul(x, x + e2)

as each iteration of the loop pays it again. The overhead can be avoided by
jitting the entire loop::

    from clifford.g3c import *
    import numba

    @numba.njit
    def mul(a, b):
        return a * b

    @numba.njit
    def the_loop(x):
        for i in range(1000):
            x = mul(x, x + e1)
        return x

    # 2.4 ms, ignoring jitting time
    x = the_loop(eq)

"""
from ._multivector import MultiVectorType
from ._layout import LayoutType
