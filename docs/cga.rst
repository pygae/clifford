
.. cga:
Conformal Geometric Algerba (CGA)
====================================

Intro
--------

Conformal Geometric Algebra (CGA) is a projective geometry tool that linearizes the conformal group. 

[ explain minkowski basis]

Mapping
--------
Vectors in the orignal space are mapped to vectors in conformal space through the map:

.. math::
    
    X = x + \frac{1}{2} x^2 e_{\infty} +e_o 


The inverse map is the made by normalizing the conformal vector, then rejection from the minkowski plane _math:`$E_0$`,


.. math:: 

    X = \frac{X}{X \cdot e_{\infty}}

then 

.. math :: 

    x = X \wedge E_0\, E_0^{-1}


Implement it 
-------------
.. ipython::

    In [138]: import clifford as cf
    
    In [138]: cf.pretty(precision=2)

    In [138]: layout, blades = cf.Cl(3,1, firstIdx=1)


Create an orthonormal basis, and a null basis 

.. ipython::

    In [138]: e1,e2,e3,e4 = [blades['e%i'%k] for k in range(1,5)]
    
    In [138]: eo = .5^(e4-e3)
    
    In [138]: einf= e3+e4
    
    In [138]: E0= einf^eo
    
    In [138]: I = e1^e2^e3^e4


Define the up and down projection functions 

.. ipython::

    In [138]: up = lambda x: x + (.5^((x**2)*einf)) + eo
   
    In [138]: homo = lambda x: x * (-x|einf).normalInv() # homogenise conformal vector
   
    In [138]: down = lambda x: (homo(x)^E0)|E0


And some methods to create vectors in the *original* 2D space, as well as the conformal 4D space. 

.. ipython::


    In [138]: v4 =  lambda : cf.randomMV(layout, grades=[1])

    In [138]: def v2():
        ....:   x=v4()
        ....:   return x- E0.project(x)
                

Some tests to sure that up-projection produces a null vector and that  up/down projection are consistent. 
.. ipython::

    In [138]:  x = v2()
    
    In [138]: X = up(x)
    
    In [138]: assert(X**2 ==0)
    
    In [138]: assert(down(X) ==x)


[ Do things...]
