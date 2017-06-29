"""
Algorithms and tools of various kinds.

Determining Rotors From Frame Pairs or Orthogonal Matrices
==========================================================

Given two frames that are related by a orthogonal transform, we seek a rotor
which enacts the transform. Details of the mathematics and psuedo-code used the
create the algorithms below can be found at Allan Cortzen's website.

 http://ctz.dk/geometric-algebra/frames-to-versor-algorithm/

There are also some helper functions which can be used to translate matrices
into GA frames, so an orthogonal (or complex unitary ) matrix can be directly
translated into a Verser.

.. autosummary::
    :toctree: generated/

    orthoFrames2Verser
    orthoMat2Verser
    mat2Frame

"""

from math import sqrt
from numpy import eye, array
from . import Cl, gp, Frame
from . import eps as global_eps


def mat2Frame(A, layout=None, is_complex=None):
    '''
    Translates a  (possibly complex) matrix into a real vector frame

    The rows and columns are interpreted as follows
        * M,N = shape(A)
        * M = dimension of space
        * N = number of vectors

    If A is complex M and N are doubled.

    Parameters
    ------------
    A : ndarray
        MxN matrix representing vectors
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
        N = N*2
        M = M*2

    if layout is None:
        layout, blades = Cl(M)

    e_ = layout.basis_vectors

    e_ = [e_['e%i' % k] for k in range(layout.firstIdx, layout.firstIdx + M)]

    a = [0 ^ e_[0]]*N

    if not is_complex:
        for n in range(N):
            for m in range(M):
                a[n] = (a[n]) + ((A[m, n]) ^ e_[m])

    else:
        for n in range(N/2):
            n_ = 2*n
            for m in range(M/2):
                m_ = 2*m

                a[n_] = (a[n_]) + ((A[m, n].real) ^ e_[m_]) \
                                + ((A[m, n].imag) ^ e_[m_+1])
                a[n_+1] = (a[n_+1]) + ((-A[m, n].imag) ^ e_[m_]) \
                                    + ((A[m, n].real) ^ e_[m_+1])
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


def orthoFrames2Verser_dist(A, B, eps=None):
    '''
    Determines verser for two frames related by an orthogonal transform

    The frames themselves do not have to be othorgonal.

    Based on [1,2]. This works  in Euclidean spaces and, under special
    circumstances in other signatures. see [1] for limitaions/details

    [1] http://ctz.dk/geometric-algebra/frames-to-versor-algorithm/

    [2] Reconstructing Rotations and Rigid Body Motions from Exact Point
    Correspondences Through Reflections, Daniel Fontijne and Leo Dorst

    '''
    # TODO: should we test to see if A and B are related by rotation?
    # TODO: implement  reflect/rotate based on distance (as in[1])

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
    dist = [abs((a - b)**2) for a, b in zip(A, B)]
    k = dist.index(max(dist))

    while dist[k] >= eps:
        r = (A[k] - B[k])/abs(A[k] - B[k])  # determine reflector
        r_list.append(r)                   # append to our list
        A = A[1:]               # remove current vector pair
        B = B[1:]

        if len(A) == 0:
            break
        # reflect remaining vectors
        for j in range(len(A)):
            A[j] = -r*A[j]*r

        # find the next pair based on current distance
        dist = [abs((a - b)**2) for a, b in zip(A, B)]
        k = dist.index(max(dist))

    # print str(len(r_list)) + ' reflections found'
    R = reduce(gp, r_list[::-1])

    return R, r_list


def orthoFrames2Verser(B, A=None, eps=None, delta=1e-3):
    '''
    Determines verser for two frames related by an orthogonal transform

    Based on [1,2]. This works  in Euclidean spaces and, under special
    circumstances in other signatures. see [1] for limitaions/details

    [1] http://ctz.dk/geometric-algebra/frames-to-versor-algorithm/

    [2] Reconstructing Rotations and Rigid Body Motions from Exact Point
    Correspondences Through Reflections, Daniel Fontijne and Leo Dorst

    '''

    # Checking and Setup
    if A is None:
        # assume we have orthonormal initial frame
        bv = B[0].layout.basis_vectors
        A = [bv[k] for k in sorted(bv.keys())]

    # make copy of original frames, so we can rotate A
    A = Frame(A[:])
    B = Frame(B[:])

    if len(A) != len(B):
        raise ValueError('len(A)!=len(B)')
    N = len(A)

    if eps is None:
        eps = global_eps()

    # Determine if we have a spinor
    spinor = False

    # compute ratio of volumes for each frame. take Nth root
    alpha = abs(B.En/A.En)**(1./N)

    if abs(alpha - 1) > eps:
        spinor = True
        # we have a spinor, remove the scaling (add it back in at the end)
        B = [b/alpha for b in B]

    # now that possible scaling has been removed, test for inner-morphism
    if not A.is_innermorphic_to(B):
        raise ValueError(
            'A and B dont appear to be related by orthogonal transform')

    # Find the Verser

    # store each reflector/rotor  in a list,  make full verser at the
    # end of the loop
    r_list = []

    for k in range(N):
        a, b = A[0], B[0]
        r = a - b                     # determine reflector
        d = abs(r**2)/abs(b**2)     # conditional rotation tolerance

        if d >= delta:
            # reflection  part
            r_list.append(r)
            A = A[1:]               # remove current vector pair
            B = B[1:]
            for j in range(len(A)):
                A[j] = -r*A[j]*r.inv()

        else:
            #  rotation part
            # if k==N:                # see paper for explaination
            #     break

            R = b*(a+b)
            r_list.append(R)       # append to our list
            A = A[1:]               # remove current vector pair
            B = B[1:]
            for j in range(len(A)):
                A[j] = R*A[j]*R.inv()

    R = reduce(gp, r_list[::-1])
    R = R/abs(R)

    if spinor:
        R = R*sqrt(alpha)

    return R, r_list


def orthoMat2Verser(A, eps=None, layout=None, is_complex=None):
    '''
    Translates an orthogonal (or unitary) matrix to a Verser

    `A` is interpreted as the frame produced by transforming a
    orthonormal frame by an orthogonal transform. Given this relation,
    this function will find the verser which enacts this transform.


    Parameters
    ------------

    '''
    B, layout = mat2Frame(A, layout=layout, is_complex=is_complex)
    N = len(B)

    # if (A.dot(A.conj().T) -eye(N/2)).max()>eps:
    #     warn('A doesnt appear to be a rotation. ')

    A, layout = mat2Frame(eye(N), layout=layout, is_complex=False)
    return orthoFrames2Verser(A=A, B=B, eps=eps)
