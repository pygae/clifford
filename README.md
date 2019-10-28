[clifford](http://clifford.readthedocs.org/en/latest/): Geometric Algebra for Python
=========================================================
A numerical geometric algebra module for python. BSD License. 

[![Build Status](https://travis-ci.org/pygae/clifford.svg?branch=master)](https://travis-ci.org/pygae/clifford)
[![Build Status](https://dev.azure.com/hadfieldhugo/clifford/_apis/build/status/pygae.clifford?branchName=master)](https://dev.azure.com/hadfieldhugo/clifford/_build/latest?definitionId=1&branchName=master)
[![Documentation Status](https://readthedocs.org/projects/clifford/badge/?version=latest)](http://clifford.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/pygae/clifford/branch/master/graph/badge.svg)](https://codecov.io/gh/pygae/clifford)
[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/pygae/clifford/master?filepath=examples%2Fg3c.ipynb) 

[![DOI](https://zenodo.org/badge/26588915.svg)](https://zenodo.org/badge/latestdoi/26588915)




Quickstart
--------------

Try out a notebook in [binder](https://mybinder.org/v2/gh/pygae/clifford/master?filepath=examples%2Fg3c.ipynb)

Or have a go on your own pc:

    from clifford.g3 import *  # import GA for 3D space
    from numpy import e,pi
    a = e1 + 2*e2 + 3*e3 # vector 
    R = e**(pi/4*e12)    # rotor 
    R*a*~R    # rotate the vector  

Summary
----------


This module implements Geometric Algebras (a.k.a. Clifford algebras). Geometric Algebra (GA) is a universal algebra which subsumes complex algebra, quaternions, linear algebra and several other independent mathematical systems.  Scalars, vectors, and higher-grade entities can
be mixed freely and consistently in the form of mixed-grade multivectors. Like this, 

![blades](https://github.com/arsenovic/clifford/blob/master/docs/_static/blades.png)


Docs
----------

[Tutorials](https://clifford.readthedocs.io/en/latest/)
* [Installation](https://clifford.readthedocs.io/en/latest/Installation.html)
* [Quick Start (G2)](https://clifford.readthedocs.io/en/latest/QuickStartG2.html)
* [The Algebra Of Space (G3)](https://clifford.readthedocs.io/en/latest/TheAlgebraOfSpaceG3.html)
* [Rotations in Space: Euler Angles, Matrices, and Quaternions](https://clifford.readthedocs.io/en/latest/EulerAngles.html)
* [Conformal Geometric Algebra](https://clifford.readthedocs.io/en/latest/ConformalGeometricAlgebra.html)
* [Space Time Algebra](https://clifford.readthedocs.io/en/latest/SpaceTimeAlgebra.html)
* [Interfacing Other Mathematical Systems](https://clifford.readthedocs.io/en/latest/InterfacingOtherMathSystems.html)
* [Predefined Geometric Algebras](https://clifford.readthedocs.io/en/latest/PredefinedGAs.html)

[API](https://clifford.readthedocs.io/en/latest/api.html)
* [clifford](https://clifford.readthedocs.io/en/latest/clifford.html)
* [cga](https://clifford.readthedocs.io/en/latest/cga.html)
* [tools](https://clifford.readthedocs.io/en/latest/tools.html)


Syntax Summary
----------

| Syntax  | Operation |
|:-:|:-:|
| \| |  Symetric inner product |
| << |  Left contraction |
|  ^ | Outer product  |
| *  |  Geometric product |
| X\(i\)  |  Return the section of the multivector X of grade i |
| X\(ei\)  |  Return the section of the multivector X for which ei is the pseudo scalar |
| X\[i\]  | Return the i'th coefficient from the multivector X
| X.normal() | Return the normalised multivector so that X*~X is +- 1 |


Links
----------

If you think Geometric Algebra looks interesting and want to learn more, check out the [cambridge group page]( http://geometry.mrao.cam.ac.uk/),  and here are some great introductory textbooks!

* Geometric Algebra for Physicists, by Doran and Lasenby
* Geometric Algebra for Computer Science, by Dorst, Fontijne and Mann
* New Foundations for Classical Mechanics, by David Hestenes



