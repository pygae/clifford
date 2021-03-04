.. currentmodule:: clifford

Changelog
=========

Changes in 1.4.x
++++++++++++++++

 * Projection using :meth:`Multivector.__call__` no longer raises :exc:`ValueError`
   for grades not present in the algebra, and instead just returns zero.

 * Where possible, ``MultiVector``\ s preserve their data type in the dual, and
   the right and left complements.

 * A new multivector inverse method is available, :meth:`clifford.MultiVector.shirokov_inverse`,
   which is the arbitrary signature algorithm described
   in Theorem 4, page 16 of Dmitry Shirokov's ICCA 2020 paper :cite:`shirokov2020inverse`.

 * A new :mod:`clifford.taylor_expansions` is added to implement taylor series of various
   multivector functions, starting with common trigonometric functions. These functions are
   additionally exposed via methods on the MultiVector class itself.

 * Random functions now accept an ``rng`` keyword argument that accepts the object returned
   by :func:`numpy.random.default_rng`, for deterministic randomness.

Changes in 1.3.x
++++++++++++++++

 * Python 3.8 is officially supported. 1.2.0 was pinned to a bad numba version
   that was incompatible with 3.8.
 * A new :mod:`clifford.operator` module to contain the previously undocumented
   :func:`~clifford.operator.gp`, :func:`~clifford.operator.op`, and
   :func:`~clifford.operator.ip` helpers.
 * A new :mod:`clifford.transformations` module for linear transformations.
 * Two new :doc:`predefined-algebras`, :mod:`clifford.dpga` and :mod:`clifford.dg3c`.
 * Improvements throughout the documentation:

   * Better overall structure, visible in the docs sidebar.
   * New tutorials for :doc:`tutorials/cga/index` on visualization and applications.
   * New tutorial on :doc:`tutorials/apollonius-cga-augmented`.
   * New tutorial on :doc:`tutorials/linear-transformations`.
   * New |launch binder| links at the top of each notebook tutorial, to run
     examples from the browser.
 * Faster algebra construction. ``Cl(3)`` is now 100\ |times| faster, and
   ``Cl(6)`` is 20\ |times| faster. This is achieved by deferring product JIT
   compilation until the product is used for the first time.
 * Additional testing and assorted improvements for :mod:`clifford.tools.g3c`:

   * :func:`~clifford.tools.g3c.closest_point_on_circle_from_line` has now been implemented
     roughly following the procedure described in Appendix A of
     `Andreas Aristidou's PhD thesis <https://www.repository.cam.ac.uk/handle/1810/237554>`_.
   * :func:`~clifford.tools.g3c.closest_point_on_line_from_circle` has now also been added,
     projecting the result of :func:`~clifford.tools.g3c.closest_point_on_circle_from_line`
     to the line.
 * :func:`clifford.ugly` now results in less ugly output for
   :doc:`predefined-algebras`.


.. |launch binder| image:: https://mybinder.org/badge_logo.svg

Bugs fixed
----------
 * :meth:`MultiVector.meet` no longer produces zero erroneously.
 * ``mv[e1 + e12]`` now raises :exc:`ValueError`, rather than being interpreted
   as ``mv[e1]``.
 * :func:`~clifford.operator.ip` (the inner product) no longer performs the
   outer product.
 * :func:`Layout.parse_multivector` now throws :exc:`SyntaxError` on invalid
   input, rather than silenltly producing nonsense.
 * :func:`Layout.parse_multivector` supports basis vector names which do not
   start with e.
 * In :mod:`clifford.tools.g3c`:

   * :func:`~clifford.tools.g3c.val_midpoint_between_lines` now handles the case that
     the two lines are touching.
   * :func:`~clifford.tools.g3c.object_fitting.val_fit_circle` now correctly selects the first and
     second eigenvalue regardless of order.
   * :func:`~clifford.tools.g3c.sphere_beyond_plane` now tested and correct.
   * :func:`~clifford.tools.g3c.sphere_behind_plane` now tested and correct.
   * :func:`~clifford.tools.g3c.val_unsign_sphere` is now jitted, as it should have
     been from the start.
   * :func:`~clifford.tools.g3c.get_nearest_plane_point` correctly returns the conformal
     point rather than the 3D point.

Compatibility notes
-------------------
 * ``clifford.grades_present`` is deprecated in favor of
   :meth:`MultiVector.grades`, the latter of which now takes an ``eps`` argument.
 * ``del mv[i]`` is no longer legal, the equivalent ``mv[i] = 0`` should be used instead.
 * ``Layout.dict_to_multivector`` has been removed. It was accidentally broken
   in 1.0.5, so there is little point deprecating it.
 * :meth:`Layout.basis_names` now returns a ``list`` of ``str``, rather than a
   numpy array of ``bytes``. The result now matches the construction order, rather
   than being sorted alphabetically. The order of :meth:`Layout.metric` has
   been adjusted for consistency.
 * The ``imt_prod_mask``, ``omt_prod_mask``, and ``lcmt_prod_mask`` attributes
   of :class:`Layout` objects have been removed, as these were an unnecessary
   intermediate computation that had no need to be public.
 * Some functions in :mod:`clifford.tools.g3c` have been renamed:

   * ``closest_points_on_circles`` has been renamed to
     :func:`~clifford.tools.g3c.iterative_closest_points_on_circles`.
   * ``closest_points_circle_line`` has been renamed to
     :func:`~clifford.tools.g3c.iterative_closest_points_circle_line`.
   * ``furthest_points_on_circles`` has been renamed to
     :func:`~clifford.tools.g3c.iterative_furthest_points_on_circles`.
 * While this release is compatible with :mod:`numba` version 0.49.0, it is
   recommended to use 0.48.0 which does not emit as many warnings. See the
   :doc:`installation` instructions for how to follow this guidance.

Patch releases
--------------
* 1.3.1: Added compatibility with :mod:`numba` version 0.50.0.

Changes in 1.2.x
++++++++++++++++

 * ``layout.isconformal``, ``layout.einf``, and ``layout.eo``, which were added
   in 1.0.4, have been removed. The first can now be spelt
   ``isinstance(layout, clifford.ConformalLayout)``, and the other properties
   now exist only on :class:`ConformalLayout` objects.
 * :meth:`MultiVector.left_complement` has been added for consistency with
   :meth:`MultiVector.right_complement`.
 * A new :mod:`clifford.tools.classify` module has been added for classifying
   blades.
 * :class:`Layout` objects print slightly more cleanly in Jupyter notebooks.
 * :attr:`Layout.scalar` is now integral rather than floating point

Bugs fixed
----------
 * ``pow(mv, 0)`` gives the right result
 * ``nan`` is now printed correctly when it appears in multivectors. Previously it was hidden
 * :meth:`MultiVector.right_complement` no longer performs the left complement.
 * :meth:`MultiVector.vee` has been corrected to have the same sign as
   :meth:`MultiVector.meet`

Compatibility notes
-------------------
 * :attr:`Layout.scalar` is now integral rather than floating point, to match
   :attr:`Layout.pseudoScalar`.


Changes in 1.1.x
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

Acknowledgements
+++++++++++++++++
Konrad Hinsen fixed a few bugs in the conversion to numpy and adding some unit
tests.

.. include:: <isonum.txt>
