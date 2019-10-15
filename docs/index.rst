
clifford: Geometric Algebra for Python
=======================================


.. ipython::

    In [138]: from clifford.g3 import *  # import GA for 3D space
    
    In [138]: from numpy import e,pi
    
    In [138]: a = e1 + 2*e2 + 3*e3 # vector 
    
    In [138]: R = e**(pi/4*e12)    # rotor 
    
    In [138]: R*a*~R    # rotate the vector  

    
    
This module implements Geometric Algebras (a.k.a. Clifford algebras). Geometric Algebra (GA) is a universal algebra which subsumes complex algebra, quaternions, linear algebra and several other independent mathematical systems.  Scalars, vectors, and higher-grade entities can
be mixed freely and consistently in the form of mixed-grade multivectors. 


.. image:: _static/blades.png
   :width: 500 px
   :align: center


.. raw:: html

      <hr>
      
Docs
====

.. toctree::
    :maxdepth: 2
    
    tutorials
    


.. toctree::
    :maxdepth: 2
    
    api
    

Slide Decks 
---------------

 * `Installing <https://slides.com/hugohadfield/installing-python-and-clifford#/>`_
 * `Conformal Geometric Algeba with Python, Part 1 <https://slides.com/hugohadfield/cgapython#/>`_
 * `Conformal Geometric Algeba with Python, Part 2 <https://slides.com/hugohadfield/cgapython-3#/>`_

Videos 
-------
 * `Intro to Clifford <https://www.youtube.com/watch?v=P2ZxxoS5YD0>`_


.. raw:: html

      <hr><br><br>

Links
=======

* Symbolic geometric algebra module for python:  https://github.com/brombo/galgebra

* Cambridge GA group:   http://www.mrao.cam.ac.uk/~clifford

* David Hestenes' (The man)  website:  http://geocalc.clas.asu.edu/

* GAOnline - Conformal Geometric Algebra visualiser: http://gaonline.azurewebsites.net/


    

If you think Geometric Algebra looks interesting and want to learn more, check out the `cambridge groups page <http://geometry.mrao.cam.ac.uk/>`_,  and here are some great introductory textbooks!

* Geometric Algebra for Physicists, by Doran and Lasenby
* Geometric Algebra for Computer Science, by Dorst, Fontijne and Mann
* New Foundations for Classical Mechanics, by David Hestenes
   


