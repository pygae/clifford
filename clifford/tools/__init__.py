"""
.. currentmodule:: clifford.tools

========================================
tools (:mod:`clifford.tools`)
========================================

Algorithms and tools of various kinds.


Tools for specific ga's
-----------------------

.. autosummary::
    :toctree: generated/

    g3
    g3c


Classifying conformal GAs
-------------------------

.. autosummary::
    :toctree: generated/

    classify


Determining Rotors From Frame Pairs or Orthogonal Matrices
-----------------------------------------------------------

Given two frames that are related by a orthogonal transform, we seek a rotor
which enacts the transform. Details of the mathematics and pseudo-code used the
create the algorithms below can be found at Allan Cortzen's website,
:cite:`ctz-frames`.

There are also some helper functions which can be used to translate matrices
into GA frames, so an orthogonal (or complex unitary) matrix can be directly
translated into a Versor.

.. autosummary::
    :toctree: generated/

    orthoFrames2Versor
    orthoMat2Versor
    mat2Frame
    omoh

"""

from functools import reduce

from typing import Union, Optional, List, Tuple
from math import sqrt
from numpy import eye, array, sign, zeros, sin, arccos
import numpy as np
from .. import Cl, gp, Frame, MultiVector, Layout
from .. import eps as global_eps

from warnings import warn


def omoh(A: Union[Frame, List[MultiVector]], B: Union[Frame, List[MultiVector]]) -> np.ndarray:
    r'''
    Determines homogenization scaling for two :class:`~clifford.Frame`\ s related by a Rotor

    This is used as part of the :func:`~clifford.tools.orthoFrames2Versor` algorithm,
    when the frames are given in CGA. It is required because the model assumes,
    ``B = R*A*~R``, but if data is given in the original space, only
    ``lambda*B' == homo(B)`` is observable.

    We need to determine lambda before the Cartan-based algorithm can be used.
    The name of this function is the reverse of
    :meth:`~clifford.ConformalLayout.homo`, which is the method used to
    homogenize.

    Parameters
    --------------
    A : list of vectors, or clifford.Frame
        the set of vectors before the transform
    B : list of vectors, or clifford.Frame
        the set of vectors after the transform, and homogenzation.
        ie ``B=B/(B|einf)``


    Returns
    ---------
    out : array of floats
        weights on `B`, which produce inhomogenous versions of `B`. If
        you multiply the input `B` by `lam`, it will fulfill `B = R*A*~R`

    Examples
    ----------
    >>> lam = omoh(A, B)  # doctest: +SKIP
    >>> B_ohom = Frame([B[k]*lam[k] for k in range(len(B)])  # doctest: +SKIP
    '''
    if len(A) != len(B) or len(A) < 3:
        raise ValueError('input must be >=3 long and len(a)==len(b)')

    idx = range(len(A))
    lam = zeros(len(A))

    for i in idx:
        j, k = [p for p in idx if p != i][:2]
        lam[i] = \
            float((A[i] * A[j])(0) * (A[i] * A[k])(0) * (B[j] * B[k])(0)) / \
            float((B[i] * B[j])(0) * (B[i] * B[k])(0) * (A[j] * A[k])(0))
        lam[i] = sqrt(float(lam[i]))

    return lam


def mat2Frame(A: np.ndarray,
              layout: Optional[Layout] = None,
              is_complex: bool = None) -> Tuple[List[MultiVector], Layout]:
    '''
    Translates a (possibly complex) matrix into a real vector frame

    The rows and columns of `A` are interpreted as follows

     * ``M, N = A.shape``
     * ``M``: dimension of space
     * ``N``: number of vectors

    If A is complex M and N are doubled.

    Parameters
    ------------
    A : ndarray
        MxN matrix representing vectors

    Returns
    -------
    a : list of clifford.MultiVector
        The resulting vectors
    layout : clifford.Layout
        The layout of the vectors in ``a``.
    '''

    # TODO: could simplify this by just implementing the real case and then
    # recursively calling this for A.real, and A.imag, then combine results

    # M = dimension of space
    # N = number of vectors
    M, N = A.shape

    if is_complex is None:
        if A.dtype == 'complex':
            is_complex = True

        else:
            is_complex = False
    if is_complex:
        N = N * 2
        M = M * 2

    if layout is None:
        layout, blades = Cl(M)

    e_ = layout.basis_vectors_lst[:M]

    a = [0 ^ e_[0]] * N

    if not is_complex:
        for n in range(N):
            for m in range(M):
                a[n] = (a[n]) + ((A[m, n]) ^ e_[m])

    else:
        for n in range(N // 2):
            n_ = 2 * n
            for m in range(M // 2):
                m_ = 2 * m

                a[n_] = (a[n_]) \
                    + ((A[m, n].real) ^ e_[m_]) \
                    + ((A[m, n].imag) ^ e_[m_ + 1])
                a[n_ + 1] = (a[n_ + 1]) \
                    + ((-A[m, n].imag) ^ e_[m_]) \
                    + ((A[m, n].real) ^ e_[m_ + 1])
    return a, layout


def frame2Mat(B, A=None, is_complex=None):
    if is_complex is not None:
        raise NotImplementedError()
    if A is None:
        # assume we have orthonormal initial frame
        A = B[0].layout.basis_vectors_lst

    # you need float() due to bug in clifford
    M = [float(b | a) for b in B for a in A]
    M = array(M).reshape(len(B), len(B))


def orthoFrames2Versor_dist(A, B, eps=None):
    '''
    Determines versor for two frames related by an orthogonal transform

    The frames themselves do not have to be orthogonal.

    Based on :cite:`ctz-frames,fontijne-reconstructing`.
    This works in Euclidean spaces and, under special
    circumstances in other signatures. See :cite:`ctz-frames` for
    limitations/details.

    '''
    # TODO: should we test to see if A and B are related by rotation?
    # TODO: implement reflect/rotate based on distance (as in:cite:`ctz-frames`)

    # keep copy of original frames
    A = A[:]
    B = B[:]

    if len(A) != len(B):
        raise ValueError('len(A)!=len(B)')

    if eps is None:
        eps = global_eps()

    # store each reflector  in a list
    r_list = []

    # find the vector pair with the largest distance
    dist = [abs((a - b) ** 2) for a, b in zip(A, B)]
    k = dist.index(max(dist))

    while dist[k] >= eps:
        r = (A[k] - B[k]) / abs(A[k] - B[k])  # determine reflector
        r_list.append(r)  # append to our list
        A = A[1:]  # remove current vector pair
        B = B[1:]

        if len(A) == 0:
            break
        # reflect remaining vectors
        for j in range(len(A)):
            A[j] = -r * A[j] * r

        # find the next pair based on current distance
        dist = [abs((a - b) ** 2) for a, b in zip(A, B)]
        k = dist.index(max(dist))

    # print(str(len(r_list)) + ' reflections found')
    R = reduce(gp, r_list[::-1])

    return R, r_list


def orthoFrames2Versor(B, A=None, delta: float = 1e-3,
                       eps: Optional[float] = None,
                       det=None,
                       remove_scaling: bool = False):
    '''
    Determines versor for two frames related by an orthogonal transform

    Based on :cite:`ctz-frames,fontijne-reconstructing`.
    This works in Euclidean spaces and, under special
    circumstances in other signatures. See :cite:`ctz-frames` for
    limitations/details.

    Parameters
    -----------
    B : list of vectors, or clifford.Frame
        the set of  vectors after the transform, and homogenzation.
        ie ``B = (B/B|einf)``

    A : list of vectors, or clifford.Frame
        the set of  vectors before the transform. If `None` we assume A is
        the basis given by ``B.layout.basis_vectors_lst``.

    delta : float
        Tolerance for reflection/rotation determination. If the normalized
        distance between A[i] and B[i] is larger than delta, we use
        reflection, otherwise use rotation.

    eps: float
        Tolerance on spinor determination. if pseudoscalar of A differs
        in magnitude from pseudoscalar of B by eps, then we have spinor.
        If `None`, use the `clifford.eps()` global eps.
    det : [+1,-1,None]
        The sign of the determinant of the versor, if known. If it is
        known a-priori that the versor is a rotation vs a reflection, this
        fact might be needed to correctly append an additional reflection
        which leaves transformed points invariant. See 4.6.3 of
        :cite:`fontijne-reconstructing`.
    remove_scaling : bool
        Remove the effects of homogenzation from frame B. This is needed
        if you are working in CGA, but the input data is given in the
        original space. See :func:`~clifford.tools.omoh` for more. See 4.6.2 of
        :cite:`fontijne-reconstructing`.

    Returns
    ---------
    R : clifford.MultiVector
        the Versor.
    rs : list of clifford.MultiVector
        ordered list of found reflectors/rotors.

    '''

    # Checking and Setup
    if A is None:
        # assume we have orthonormal initial frame
        A = B[0].layout.basis_vectors_lst

    # make copy of original frames, so we can rotate A
    A = A[:]
    B = B[:]

    if len(A) != len(B):
        raise ValueError('len(A)!=len(B)')

    if eps is None:
        eps = global_eps()

    # Determine if we have a spinor
    spinor = False
    # store pseudoscalar of frame B, in case known det (see end)
    try:
        B = Frame(B)
        B_En = B.En
    except Exception:
        pass
    N = len(A)

    # Determine and remove scaling factors caused by homogenization
    if remove_scaling:
        lam = omoh(A, B)
        B = Frame([B[k] * lam[k] for k in range(N)])

    try:
        # compute ratio of volumes for each frame. take Nth root
        A = Frame(A[:])
        B = Frame(B[:])
        alpha = abs(B.En / A.En) ** (1. / N)

        if abs(alpha - 1) > eps:
            spinor = True
            # we have a spinor, remove the scaling (add it back in at the end)
            B = [b / alpha for b in B]
    except Exception:
        # probably  A and B are not pure vector correspondence
        # whatever,  it might still work
        pass

    # Find the Versor

    # store each reflector/rotor in a list, make full versor at the
    # end of the loop
    r_list = []

    for k in range(N):
        a, b = A[0], B[0]
        r = a - b  # determine reflector
        if abs(b ** 2) > eps:
            d = abs(r ** 2) / abs(b ** 2)  # conditional rotation tolerance
        else:
            # probably b is a null vector, make our best guess for tol!
            d = abs(r ** 2)

        if d >= delta:
            # reflection  part
            r_list.append(r)
            A = A[1:]  # remove current vector pair
            B = B[1:]
            for j in range(len(A)):
                A[j] = -r * A[j] * r.inv()

        else:
            #  rotation part
            # if k==N:                # see paper for explanation
            #     break

            R = b * (a + b)
            if abs(R) > eps:  # abs(R) can be <eps in null space
                r_list.append(R)  # append to our list
            A = A[1:]  # remove current vector pair
            B = B[1:]
            for j in range(len(A)):
                A[j] = R * A[j] * R.inv()

    R = reduce(gp, r_list[::-1])

    # if det is known a priori check to see if it's correct, if not add
    # an extra reflection which leaves all points in B invariant
    if det is not None:
        I = R.pseudoScalar
        our_det = (R * I * ~R * I.inv())(0)
        if sign(float(our_det)) != det:
            R = B_En.dual() * R

    if abs(R) < eps:
        warn('abs(R)<eps. likely to be inaccurate')
    R = R / abs(R)

    if spinor:
        R = R * sqrt(alpha)

    return R, r_list


def orthoMat2Versor(A, eps=None, layout=None, is_complex=None):
    '''
    Translates an orthogonal (or unitary) matrix to a Versor

    `A` is interpreted as the frame produced by transforming a
    orthonormal frame by an orthogonal transform. Given this relation,
    this function will find the versor which enacts this transform.

    Parameters
    ------------

    '''
    B, layout = mat2Frame(A, layout=layout, is_complex=is_complex)
    N = len(B)

    # if (A.dot(A.conj().T) -eye(N/2)).max()>eps:
    #     warn('A doesnt appear to be a rotation. ')
    A, layout = mat2Frame(eye(N), layout=layout, is_complex=False)
    return orthoFrames2Versor(A=A, B=B, eps=eps)


def rotor_decomp(V: MultiVector, x: MultiVector) -> Tuple[MultiVector, MultiVector]:
    '''
    Rotor decomposition of rotor V

    Given a rotor V, and a vector x, this will decompose V into a
    series of two rotations, U and H, where U leaves x
    invariant and H contains x.

    Limited to 4D for now.

    See :cite:`hestenes-space-time`, Appendix B, Theorem 4.

    Parameters
    ---------------
    V : clifford.MultiVector
        rotor
    x : clifford.MultiVector
        vector

    Returns
    -------
    H : clifford.Multivector
        rotor which contains x
    U : clifford.Multivector
        rotor which leaves x invariant

    '''
    H2 = V * x * ~V * x.inv()  # inv needed to handle signatures
    H = (1 + H2) / sqrt(abs(float(2 * (1 + H2(0)))))
    U = H * x * V * x.inv()
    return H, U


def sinc(x):
    return sin(x)/x


def log_rotor(V):
    '''
    Logarithm of a simple rotor
    '''
    if (V(2)**2).grades() != {0}:
        print(V)
        # raise ValueError('Bivector is not a Blade.')
    if abs(V(2)) < global_eps():
        return log(float(V(0)))
    # numpy's trig correctly chooses hyperbolic or not with Complex args
    theta = arccos(complex(V(0)))
    return V(2)/sinc(theta).real
