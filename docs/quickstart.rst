
.. _quickstart:
QuickStart (G2)
================

Instantiate a G2 algebra 

.. ipython::

    In [138]:import clifford as cf

    In [138]:cf.pretty(precision=2)    # sets display precision 

    In [138]:layout, blades = cf.Cl(2) # creates a 2-dimensional clifford algebra

Inspect Blades

.. ipython::

    In [138]: blades

Assign Blades

.. ipython::

    In [138]: e0 = blades['e0']
   
    In [138]: e1 = blades['e1']
    
    In [138]: e01 = blades['e01']
    
    
Basics

.. ipython::

    In [4]:e0*e1 # geometric product

    In [5]:e0^e1 # outer product

    In [6]:e0|e1 # inner product

Rotation

.. ipython::

    In [138]:  from scipy.constants import e,pi

    In [138]:  R = e**(pi/4 *e01)
    
    In [138]:  R*e0*~R
    
    
Reflection

.. ipython::
   
    In [138]:  a = e0+e1
    
    In [138]:  n = e1
    
    In [138]:  -n*a*n # reflect in hyperplane normal to `n`
    
    
