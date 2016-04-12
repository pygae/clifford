from warnings import warn
from . import Cl, gp

def mat2Frame(A, layout=None):
    '''
    Translates a [complex] matrix into a real frame
    
    The rows and columns are interpreted as follows
        * M,N = shape(A)
        * M = dimension of space
        * N = number of vectors
        
    If A is complex M and N are doubled. If M>6, this is very slow. 
    
    Parameters 
    ------------
    A : ndarray
        MxN matrix representing vectors 
    '''
    
    ## TODO: could simplify this by just implementing the real case 
    ## and then recursively calling this for A.real, and A.imag, then 
    ## combine results
    
    # M = dimension of space
    # N = number of vectors
    M,N = A.shape
    
    if A.dtype == 'complex':
        is_complex = True
        N = N*2
        M = M*2
        
    else:
        is_complex = False
    
    if layout is None:
        layout, blades = Cl(M,firstIdx=0)
        
    e_ = layout.basis_vectors()
    e_ = [e_['e%i'%k] for k in range(M)]
    
    a=[0^e_[0]]*N
    
    if not is_complex:
        for n in range(N):
            for m in range(M):
                a[n] = (a[n]) + ((A[m,n])^e_[m])
                
    else:
        for n in range(N/2):
            n_ = 2*n
            for m in range(M/2):
                m_ = 2*m
                
                a[n_] = (a[n_]) + ((A[m,n].real)^e_[m_]) \
                                + ((A[m,n].imag)^e_[m_+1])
                a[n_+1] = (a[n_+1]) + ((-A[m,n].imag)^e_[m_]) \
                                    + ((A[m,n].real)^e_[m_+1])
        
        
        
    return a, layout

def orthoFrames2Verser2(A,B, eps =1e-6):
    '''
    Determines verser for two frames related by an orthogonal tfrm
    
    
    '''
    A = A[:] # keep copy of original frame
    ## TODO: shoudl we test to see if A and B are related by rotation?
    if len(A)!=len(B):
        raise ValueError('len(A)!=len(B)')
    
    N = len(A)
    
    
    # store each reflector  in a list 
    rs = [1]*N
    
    for k in range(N):
        if abs((A[k]*B[k])-1) <eps:
            continue
        r = (A[k]-B[k])/abs(A[k]-B[k])
        for j  in range(k,N):
            A[j] = -r&A[j]&r

        rs[k] =r
    
    
    R = reduce(gp,rs[::-1] )
    #A=A_
    return R,rs

def orthoFrames2Verser(A,B, eps =1e-6):
    '''
    Determines verser for two frames related by an orthogonal transform
    
    Based on [1]. This works  in Euclidean spaces and, under special 
    circumstances in other signatures. see [1] for limitaions/details
    
    
    [1] Reconstructing Rotations and Rigid Body Motions from Exact Point 
    Correspondences Through Reflections, Daniel Fontijne and Leo Dorst
    
    '''
    ## TODO: shoudl we test to see if A and B are related by rotation?
    
    # keep copy of original frames
    A = A[:] 
    B = B[:]
    
    
    if len(A)!=len(B):
        raise ValueError('len(A)!=len(B)')
    
    N = len(A)
        
    # store each reflector  in a list 
    r_list = []
    
    # find the vector pair with the largest distance
    dist = [abs((a-b)**2) for a,b in zip(A,B)]
    k =  dist.index(max(dist))
    
    while dist[k] >= eps:
        r = (A[k]-B[k])/abs(A[k]-B[k]) # determine reflector
        r_list.append(r)                   # append to our list 
        A.pop(k)                       # remove current vector pair
        B.pop(k)

        if len(A)==0:
            break
        # reflect remaining vectors 
        for j in range(len(A)):
            A[j] = -r&A[j]&r
       
        # find the next pair based on current distance
        dist = [abs((a-b)**2) for a,b in zip(A,B)]
        k = dist.index(max(dist))
    
    print str(len(r_list)) + ' reflections found'
    R = reduce(gp,r_list[::-1] )
    
    return R ,r_list

def orthoMat2Verser(A, eps= 1e-6,layout=None):
    '''
    Translates a [complex] orthogonal matrix to a Verser 
    '''
    B,layout = Mat2Frame(A,layout=layout)
    N = len(B)
    A,layout = Mat2Frame(eye(N),layout=layout)
    
    if (A.dot(A.conj().T) -eye(N)).max()>eps:
            warn('A doesnt appear to be a rotation. ')

    return orthoFramePair2Verser(A,B, eps = eps)
