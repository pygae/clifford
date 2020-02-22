
clifford: Geometric Algebra for Python
=======================================


.. ipython::
    :doctest:

    In [138]: from clifford.g3 import *  # import GA for 3D space

    In [138]: import math

    In [138]: a = e1 + 2*e2 + 3*e3  # vector

    In [138]: R = math.e**(math.pi/4*e12)    # rotor

    In [138]: R*a*~R    # rotate the vector
    Out[138]: (2.0^e1) - (1.0^e2) + (3.0^e3)


This module implements Geometric Algebras (a.k.a. Clifford algebras).
Geometric Algebra (GA) is a universal algebra which subsumes complex algebra, quaternions, linear algebra and several other independent mathematical systems.
Scalars, vectors, and higher-grade entities can be mixed freely and consistently in the form of mixed-grade multivectors.


.. image:: _static/blades.png
   :width: 500 px
   :align: center


.. raw:: html

      <hr>

.. toctree::
    :maxdepth: 2

    tutorials



.. toctree::
    :maxdepth: 2

    api

Other resources for ``clifford``
--------------------------------

Slide Decks
~~~~~~~~~~~

 * `Installation <https://slides.com/hugohadfield/installing-python-and-clifford#/>`_
 * `Conformal Geometric Algebra with Python, Part 1 <https://slides.com/hugohadfield/cgapython#/>`_
 * `Conformal Geometric Algebra with Python, Part 2 <https://slides.com/hugohadfield/cgapython-3#/>`_

Videos
~~~~~~
 * `Intro to Clifford <https://www.youtube.com/watch?v=P2ZxxoS5YD0>`_


Other resources for Geometric Algebra
-------------------------------------

If you think Geometric Algebra looks interesting and want to learn more, check out these websites and textbooks

Links
~~~~~

* |galgebra|_
* `The Cambridge University Geometric Algebra Research Group home page <http://www.mrao.cam.ac.uk/~clifford>`_
* David Hestenes' `Geometric Calculus R & D Home Page <http://geocalc.clas.asu.edu/>`_

.. |galgebra| replace:: ``galgebra``, a symbolic geometric algebra module for Python
.. _galgebra: https://github.com/pygae/galgebra

Introductory textbooks
~~~~~~~~~~~~~~~~~~~~~~

* `Geometric Algebra for Physicists <ga4physicists>`_, by Doran and Lasenby
* `Geometric Algebra for Computer Science <ga4cs>`_, by Dorst, Fontijne and Mann
* `New Foundations for Classical Mechanics <nf4mechanics>`_, by David Hestenes

.. _ga4physicists: http://www.mrao.cam.ac.uk/~cjld1/pages/book.htm
.. _ga4cs: https://dl.acm.org/doi/book/10.5555/1610323
.. _nf4mechanics: http://geocalc.clas.asu.edu/html/NFCM.html
