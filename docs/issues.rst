
.. currentmodule:: clifford

Issues
======

.. warning::

    This document is kept for historic reasons, but may no longer reflect the current
    state of the latest release of ``clifford``.
    For the most up to date source of issues, look at `the GitHub issue tracker <https://github.com/pygae/clifford/issues>`_.



* Currently, algebras over 6 dimensions are very slow. this is because
  this module was written for *pedagogical* purposes. However, because the
  syntax for this module is so attractive, we plan to fix the
  performance problems,  in the future...

* Due to Python's :ref:`order of operations <python:operator-summary>`, the bit operators ``^`` ``<<`` ``|``
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



--------

Happy hacking!

Robert Kern

robert.kern@gmail.com
