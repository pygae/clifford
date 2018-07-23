Issues
======

 * Currently, algebras over 6 dimensions are very slow. this is because
   this module was written for *pedagogical* purposes. However, because the
   syntax for this module is so attractive, we plan to fix the
   perfomance problems,  in the future...

 * Due to Python's order of operations, the bit operators ^ << follow
   the normal arithmetic operators + - * /, so

     1^e0 + 2^e1  !=  (1^e0) + (2^e1)

   as is probably intended.  Additionally,

     M = MultiVector(layout2D)  # null multivector
     M << 1^e0 << 2^e1 == 10.0^e1 + 1.0^e01
     M == 1.0
     e0 == 2 + 1^e0

   as is definitely not intended.  However,

     M = MultiVector(layout2D)
     M << (2^e0) << e1 << (3^e01) == M == 2^e0 + 1^e1 + 3^e01
     e0 == 1^e0
     e1 == 1^e1
     e01 == 1^e01

 * Since * is the inner product and the inner product with a scalar
   vanishes by definition, an expression like

     1|e0 + 2|e1

   is null.  Use the outer product or full geometric product, to
   multiply scalars with MultiVectors.  This can cause problems if
   one has code that mixes Python numbers and MultiVectors.  If the
   code multiplies two values that can each be either type without
   checking, one can run into problems as "1 | 2" has a very different
   result from the same multiplication with scalar MultiVectors.

 * Taking the inverse of a MultiVector will use a method proposed by
   Christian Perwass that involves the solution of a matrix equation.
   A description of that method follows:

   Representing multivectors as 2**dims vectors (in the matrix sense),
   we can carry out the geometric product with a multiplication table.
   In pseudo-tensorish language (using summation notation):

     m_i * g_ijk * n_k = v_j

   Suppose m_i are known (M is the vector we are taking the inverse of),
   the g_ijk have been computed for this algebra, and v_j = 1 if the
   j'th element is the scalar element and 0 otherwise, we can compute the
   dot product m_i * g_ijk.  This yields a rank-2 matrix.  We can
   then use well-established computational linear algebra techniques
   to solve this matrix equation for n_k.  The laInv method does precisely
   that.

   The usual, analytic, method for computing inverses [M**-1 = ~M/(M*~M) iff
   M*~M == |M|**2] fails for those multivectors where M*~M is not a scalar.
   It is only used if the inv method is manually set to point to normalInv.

   My testing suggests that laInv works.  In the cases where normalInv works,
   laInv returns the same result (within _eps).  In all cases,
   M * M.laInv() == 1.0 (within _eps).  Use whichever you feel comfortable
   with.

   Of course, a new issue arises with this method.  The inverses found
   are sometimes dependant on the order of multiplication.  That is:

     M.laInv() * M == 1.0
     M * M.laInv() != 1.0

   XXX Thus, there are two other methods defined, leftInv and rightInv which
   point to leftLaInv and rightLaInv.  The method inv points to rightInv.
   Should the user choose, leftInv and rightInv will both point to normalInv,
   which yields a left- and right-inverse that are the same should either exist
   (the proof is fairly simple).

 * The basis vectors of any algebra will be orthonormal unless you supply
   your own multiplication tables (which you are free to do after the Layout
   constructor is called).  A derived class could be made to calculate these
   tables for you (and include methods for generating reciprocal bases and the
   like).

 * No care is taken to preserve the dtype of the arrays.  The purpose
   of this module is pedagogical.  If your application requires so many
   multivectors that storage becomes important, the class structure here
   is unsuitable for you anyways.  Instead, use the algorithms from this
   module and implement application-specific data structures.

 * Conversely, explicit typecasting is rare.  MultiVectors will have
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

Changes 0.6-0.7
+++++++++++++++++

 * Added a real license.
 * Convert to numpy instead of Numeric.

Changes 0.5-0.6
+++++++++++++++++

 * join() and meet() actually work now, but have numerical accuracy problems
 * added clean() to MultiVector
 * added leftInv() and rightInv() to MultiVector
 * moved pseudoScalar() and invPS() to MultiVector (so we can derive
   new classes from MultiVector)
 * changed all of the instances of creating a new MultiVector to create
   an instance of self.__class__ for proper inheritance
 * fixed bug in laInv()
 * fixed the massive confusion about how dot() works
 * added left-contraction
 * fixed embarassing bug in gmt generation
 * added normal() and anticommutator() methods
 * fixed dumb bug in elements() that limited it to 4 dimensions

Happy hacking!

Robert Kern

robert.kern@gmail.com
