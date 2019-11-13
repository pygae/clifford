
.. currentmodule:: clifford

Issues
======


 * Currently, algebras over 6 dimensions are very slow. this is because
   this module was written for *pedagogical* purposes. However, because the
   syntax for this module is so attractive, we plan to fix the
   performance problems,  in the future...

 * Due to Python's `order of operations`_, the bit operators ``^`` ``<<`` ``|``
   are evaluated after the normal arithmetic operators ``+`` ``-`` ``*`` ``/``,
   which do not follow the precedence expected in GA ::

     # written        meaning            possibly intended
     1^e1 + 2^e2   == 1^(e1+2)^e2     != (1^e0) + (2^e1)
     e2 + e1|e2    == (e2 + e1)|e2    != e1 + (e1|e2)

   This can also cause confusion within the bitwise operators::

     # written        meaning            possibly intended
     e1 << e2 ^ e1 == (e1 << e2) ^ e1 != e1 << (e2 ^ e1)
     e1 ^ e2 | e1  == (e1 << e2) ^ e1 != e1 << (e2 ^ e1)

 * Since ``|`` is the inner product and the inner product with a scalar
   vanishes by definition, an expression like::

     (1|e0) + (2|e1)

   is null.  Use the outer product or full geometric product, to
   multiply scalars with :class:`MultiVector`\ s.  This can cause problems if
   one has code that mixes Python numbers and MultiVectors.  If the
   code multiplies two values that can each be either type without
   checking, one can run into problems as ``1 | 2`` has a very different
   result from the same multiplication with scalar MultiVectors.

 * Taking the inverse of a :class:`MultiVector` will use a method proposed by
   Christian Perwass that involves the solution of a matrix equation.
   A description of that method follows:

   Representing multivectors as :math:`2^\text{dims}`-vectors (in the matrix sense),
   we can carry out the geometric product with a multiplication table.
   In pseudo-tensorish language (using summation notation)

   .. math::

     m_i g_{ijk} n_k = v_j

   Suppose :math:`m_i` are known (M is the vector we are taking the inverse of),
   the :math:`g_{ijk}` have been computed for this algebra, and :math:`v_j = 1`
   if the :math:`j`'th element is the scalar element and 0 otherwise, we can compute the
   dot product :math:`m_i g_{ijk}`.  This yields a rank-2 matrix.  We can
   then use well-established computational linear algebra techniques
   to solve this matrix equation for :math:`n_k`.  The ``laInv`` method does precisely
   that.

   The usual, analytic, method for computing inverses (:math:`M^{-1} = \tilde M/(M \tilde M)`
   iff :math:`M\tilde M = {|M|}^2``) fails for those multivectors where ``M*~M`` is not a scalar.
   It is onl)y used if the ``inv`` method is manually set to point to ``normalInv``.

   My testing suggests that ``laInv`` works.  In the cases where ``normalInv`` works,
   ``laInv`` returns the same result (within ``_eps``).  In all cases,
   ``M * M.laInv() == 1.0`` (within ``_eps``).  Use whichever you feel comfortable
   with.

   Of course, a new issue arises with this method.  The inverses found
   are sometimes dependant on the order of multiplication.  That is::

     M.laInv() * M == 1.0
     M * M.laInv() != 1.0

   XXX Thus, there are two other methods defined, ``leftInv`` and ``rightInv`` which
   point to ``leftLaInv`` and ``rightLaInv``.  The method ``inv`` points to ``rightInv``.
   Should the user choose, ``leftInv`` and ``rightInv`` will both point to ``normalInv``,
   which yields a left- and right-inverse that are the same should either exist
   (the proof is fairly simple).

 * The basis vectors of any algebra will be orthonormal unless you supply
   your own multiplication tables (which you are free to do after the :class:`Layout`
   constructor is called).  A derived class could be made to calculate these
   tables for you (and include methods for generating reciprocal bases and the
   like).

 * No care is taken to preserve the dtype of the arrays.  The purpose
   of this module is pedagogical.  If your application requires so many
   multivectors that storage becomes important, the class structure here
   is unsuitable for you anyways.  Instead, use the algorithms from this
   module and implement application-specific data structures.

 * Conversely, explicit typecasting is rare. :class:`MultiVector`\ s will have
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

Upcoming changes
++++++++++++++++

 * ``layout.isconformal``, ``layout.einf``, and ``layout.eo``, which were added
   in 1.0.4, have been removed. The first can now be spelt
   ``isinstance(layout, clifford.ConformalLayout)``, and the other properties
   now exist only on :class:`ConformalLayout` objects.


Changes in 1.1.0
++++++++++++++++

 * Restores ``layout.gmt``, ``Layout.omt``, ``Layout.imt``, and ``Layout.lcmt``.
   A few releases ago, these existed but were dense.
   For memory reasons, they were then removed entirely.
   They have now been reinstated as :class:`sparse.COO` matrix objects, which
   behave much the same as the original dense arrays.


 * ``MultiVector``\ s preserve their data type in addition, subtraction, and
   products. This means that integers remain integers until combined with
   floats. Note that this means in principle integer overflow is possible, so
   working with floats is still recommended. This also adds support for floating
   point types of other precision, such as ``np.float32``.

 * ``setup.py`` is now configured such that ``pip2 install clifford`` will not
   attempt to download this version, since it does not work at all on python 2.

 * Documentation now includes examples of ``pyganja`` visualizations.

Compatibility notes
-------------------

 * ``Layout.blades()`` now includes the scalar ``1``, as do other similar
   functions.

 * ``MultiVector.grades()`` now returns a :class:`set` not a :class:`list`.
   This means code like ``mv.grades() == [0]`` will need to change to
   ``mv.grades() == {0}``, or to work both before and after this change,
   ``set(mv.grades()) == {0}``.

Bugs fixed
----------

 * ``mv[(i, j)]`` would sometimes fail if the indices were not in canonical order.
 * ``mv == None`` and ``layout == None`` would crash rather than return ``False``.
 * ``blade.isVersor()`` would return ``False``.
 * ``layout.blades_of_grade(0)`` would not return the list it claimed to return.

Internal changes
----------------

 * Switch to ``pytest`` for testing.
 * Enable code coverage.
 * Split into smaller files.
 * Remove python 2 compatibility code, which already no longer worked.


Changes 0.6-0.7
+++++++++++++++++

 * Added a real license.
 * Convert to NumPy instead of Numeric.

Changes 0.5-0.6
+++++++++++++++++

 * ``join()`` and ``meet()`` actually work now, but have numerical accuracy problems
 * added ``clean()`` to :class:`MultiVector`
 * added ``leftInv()`` and ``rightInv()`` to :class:`MultiVector`
 * moved ``pseudoScalar()`` and ``invPS()`` to :class:`MultiVector` (so we can derive
   new classes from :class:`MultiVector`)
 * changed all of the instances of creating a new MultiVector to create
   an instance of ``self.__class__`` for proper inheritance
 * fixed bug in laInv()
 * fixed the massive confusion about how dot() works
 * added left-contraction
 * fixed embarrassing bug in gmt generation
 * added ``normal()`` and ``anticommutator()`` methods
 * fixed dumb bug in :func:`elements()` that limited it to 4 dimensions

Happy hacking!

Robert Kern

robert.kern@gmail.com


.. _`order of operations`: https://docs.python.org/3/reference/expressions.html#operator-precedence