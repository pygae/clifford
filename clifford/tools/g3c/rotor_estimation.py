import random
from scipy import e
import numpy as np
import multiprocessing

from scipy.optimize import minimize

from .rotor_parameterisation import rotorconversion
from . import rotor_between_objects, apply_rotor, square_roots_of_rotor, rotor_between_lines, normalised
from clifford.g3c import *
from clifford.tools import orthoFrames2Verser as cartan
from clifford.tools.g3c import *
from .cost_functions import object_set_cost_sum, rotor_cost
from clifford import grade_obj, MVArray

I3 = e123
I5 = e12345
imt_func = layout.imt_func
gmt_func = layout.gmt_func
inv_func = layout.inv_func
adjoint_func = layout.adjoint_func
e4_val = e4.value
ninf_val = einf.value
e123inf =e123*einf

motor_basis = [1 + 0 * e1, e12, e13, e23,
               e1 * einf, e2 * einf, e3 * einf,
               I3 * einf]


def extract_rotor_from_TRS_mat_est(mat_est):
    """
    Given a matrix of the form [TRS_left@~TRS_right] returns TRS
    """
    sph = (up(e1)^up(-e1)^up(e2)^up(e3)).normal()*I5
    sph2 = layout.MultiVector(value=mat_est@sph.value).normal()
    t = down((sph2*einf*sph2)(1))
    T = generate_translation_rotor(t)
    S = generate_dilation_rotor(get_radius_from_sphere(sph2*I5)/get_radius_from_sphere(sph*I5))
    TS = T*S
    Rest = 1 + sum((1/ layout.MultiVector( value=val_apply_rotor(mat_est@ebase.value,(~TS).value) ))*ebase for ebase in [e1,e2,e3])
    return (TS*Rest).normal()


def lambda_estimate(Y_point_list, X_point_list, n_samples=50):
    """
    Does a sample based lambda estimate as per Alex Arsenovic
    """
    lamb2est = 0
    for m in range(n_samples):
        indices = random.sample(range(len(X_point_list)), k=3)
        i = indices[0]
        j = indices[1]
        k = indices[2]
        lambdai2 = (X_point_list[i]|X_point_list[j])[0]
        lambdai2*=(X_point_list[i]|X_point_list[k])[0]
        lambdai2*=(Y_point_list[j]|Y_point_list[k])[0]
        denom = (Y_point_list[i]|Y_point_list[j])[0]
        denom*= (Y_point_list[i]|Y_point_list[k])[0]
        denom*= (X_point_list[j]|X_point_list[k])[0]
        lamb2est += lambdai2/denom
    return np.sqrt(lamb2est/n_samples)


def direct_TRS_extraction(Y, X):
    """
    Directly extracts the TRS rotor from vector correspondences
    """
    l = lambda_estimate(Y, X, n_samples=100)
    XArray = np.array([x.value for x in X]).T
    YArray = np.array([y.value for y in Y]).T * l
    lstsol = np.linalg.lstsq(XArray.T, YArray.T, rcond=None)[0].T
    est_rotor = extract_rotor_from_TRS_mat_est(lstsol)
    return est_rotor


@numba.njit
def val_project_bv(bv, x):
    return -imt_func(imt_func(x,bv),bv)


@numba.njit
def val_calc_error(R, A, Y):
    sumout = 0
    for i in range(A.shape[0]):
        sumout += val_norm(Y[i,:] - val_apply_rotor(A[i,:], R))
    return sumout/A.shape[0]


@numba.njit
def val_in_plane_estimate_rotation(bv, A, Y):
    rightside = np.zeros(32)
    leftside = np.zeros(32)
    for i in range(A.shape[0]):
        Abv = val_project_bv(bv, A[i,:])
        Ybv = val_project_bv(bv, Y[i,:])
        revAbv = adjoint_func(Abv)
        rightside += gmt_func(revAbv, Ybv)
        leftside += gmt_func(revAbv,Abv)
    R = gmt_func(inv_func(leftside),rightside)
    R[0] += 1
    return adjoint_func(val_normalised(R))

e23_val = e23.value
e12_val = e12.value
e13_val = e13.value
dekeninckbivmat = np.array([e23_val, e12_val, e13_val])

@numba.njit
def val_de_keninck_twist(Y, X, guess):
    """
    Performs the De Keninck twist
    Estimates the rotation between vectors
    """
    lasterr = 1E20
    nmvs = X.shape[0]
    A = np.zeros((nmvs,32))
    for steps in range(1, 100):
        for i in range(nmvs):
            A[i,:] = val_apply_rotor(X[i,:], guess)
        biv = dekeninckbivmat[steps % 3, :]
        newguess = val_normalised(gmt_func(val_in_plane_estimate_rotation(biv, A, Y), guess))
        newerr = val_calc_error(newguess, X, Y)
        if newerr < lasterr:
            guess = newguess
        if abs(lasterr-newerr)<1E-6:
            break
        lasterr = newerr
    return guess


def de_keninck_twist(Y, X, guess=None):
    """
    Performs the De Keninck twist
    Estimates the rotation between vectors
    """
    if guess is None:
        guess = (1 + 0*e1)
    return layout.MultiVector(value=val_de_keninck_twist(Y.value, X.value, guess.value))


def check_p(Q):
    """ For leo dorsts check product """
    return Q(0, 1, 3) - Q(2, 4, 5)


def average_estimator(reference_model, query_model):
    """
    Estimates the rotor that takes one set of objects to another
    by estimating the rotor between the average objects
    """
    r_av = sum(reference_model)(grade_obj(reference_model[0]))
    q_av = sum(query_model)(grade_obj(reference_model[0]))
    return rotor_between_objects(q_av, r_av)


def estimate_rotor_objects_subsample(reference_model, query_model, n_repeats=None, objects_per_sample=None,
                                     maxfev=20000, print_res=False, pool_size=1, object_type='generic'):
    """
    Estimates the rotor that takes one set of objects to another
    """
    if n_repeats is None:
        n_repeats = int(len(query_model)/2)
    if objects_per_sample is None:
        objects_per_sample = max(int(len(query_model)/10), 5)
    if int(pool_size) < 2:
        min_cost = np.finfo(float).max
        min_rotor = 1.0 + 0.0 * e1
        for i in range(n_repeats):
            indices = random.sample(range(len(reference_model)), objects_per_sample)
            object_sample_reference = [reference_model[j] for j in indices]
            object_sample_query = [query_model[j] for j in indices]
            rotor, new_cost = estimate_rotor_objects(object_sample_reference, object_sample_query, object_type=object_type)
            if new_cost < min_cost:
                min_cost = new_cost
                min_rotor = rotor
            #print('SAMPLE: ', i, '  cost  ', min_cost)
    elif int(pool_size) > 1:
        with multiprocessing.Pool(int(pool_size)) as pool_obj:
            object_sample_pairs = []
            for i in range(n_repeats):
                indices = random.sample(range(len(reference_model)), objects_per_sample)
                object_sample_reference = [reference_model[j] for j in indices]
                object_sample_query = [query_model[j] for j in indices]
                object_sample_pairs.append([object_sample_reference, object_sample_query])
            if object_type == 'lines':
                starmap_output = pool_obj.starmap(estimate_rotor_lines, object_sample_pairs)
            else:
                starmap_output = pool_obj.starmap(estimate_rotor_objects, object_sample_pairs)
            min_rotor, min_cost = min(starmap_output, key=lambda x: x[1])
            #print('SAMPLE: ', n_repeats, '  cost  ', min_cost)
    else:
        raise ValueError('Pool size is not valid')
    return min_rotor, min_cost


def dorst_motor_estimate(Q_in, P_in):
    """
    Estimating rotors from a variety of data
    """
    # Form linear operator
    Q_list = [check_p(Q) for Q in Q_in]
    P_list = P_in
    lag_mat = np.zeros((32, 32))
    n = len(Q_in)
    wk = 1 / n
    for k in range(n):
        Qk = Q_list[k]
        Qkrev = ~Qk
        Pk = P_list[k]
        Pkrev = ~Pk
        Qk_left = get_left_gmt_matrix(Qk.value)
        Pk_right = get_right_gmt_matrix(Pk.value)
        Qk_left_rev = get_left_gmt_matrix(Qkrev.value)
        Pk_right_rev = get_right_gmt_matrix(Pkrev.value)
        lag_mat += 0.5 * wk * (Qk_left @ Pk_right + Qk_left_rev @ Pk_right_rev)
    # Steps 2 and 3
    L = np.zeros((8, 8))
    for i in range(8):
        for j in range(8):
            L[i, j] = gmt_func((~motor_basis[i]).value, lag_mat @ motor_basis[j].value)[0]
    Lrr = L[0:4, 0:4]
    Lrq = L[0:4, 4:8]
    Lqr = L[4:8, 0:4]
    Lqq = L[4:8, 4:8]
    Lqqplus = np.linalg.pinv(Lqq)
    Ldash = Lrr - Lrq @ (Lqqplus @ Lqr)

    # Step 4
    w, v = np.linalg.eig(Ldash)
    max_e_val = 0.0
    n = 0
    for i in range(4):
        this_e_val = w[i]
        if this_e_val > max_e_val:
            max_e_val = this_e_val
            n = i
    r = v[:, n]

    # Step 5
    q = -((Lqqplus @ Lqr) @ r)
    # Step 6
    m = np.zeros(8)
    m[0:4] = r
    m[4:8] = q
    # Step 7
    M = sum((m[k] * motor_basis[k] for k in range(8))).normal()
    return M


def estimate_rotor_objects_subsample_sequential(reference_model, query_model, n_repeats=None, objects_per_sample=None,
                                     maxfev=20000, print_res=False, pool_size=1, object_type='generic'):
    """
    Estimates the rotor that takes one set of objects to another
    """
    if n_repeats is None:
        n_repeats = int(len(query_model)/2)
    if objects_per_sample is None:
        objects_per_sample = max(int(len(query_model)/10), 5)
    min_cost = np.finfo(float).max
    min_rotor = 1.0 + 0.0 * e1
    for i in range(n_repeats):
        indices = random.sample(range(len(reference_model)), objects_per_sample)
        object_sample_reference = [reference_model[j] for j in indices]
        object_sample_query = [query_model[j] for j in indices]
        rotor, e_flag = sequential_object_rotor_estimation(object_sample_reference, object_sample_query,
                                                             object_type=object_type)
        query_model_remapped = [normalised(apply_rotor(l, rotor)) for l in object_sample_query]
        new_cost = object_set_cost_sum(reference_model, query_model_remapped, object_type=object_type)
        if new_cost < min_cost:
            min_cost = new_cost
            min_rotor = rotor
        #print('SAMPLE: ', i, '  cost  ', min_cost)
    return min_rotor, min_cost



def estimate_rotor_lines(reference_model, query_model, maxfev=20000, print_res=False):
    return estimate_rotor_objects(reference_model, query_model, maxfev=maxfev, print_res=print_res, object_type='lines')


def estimate_rotor_objects(reference_model, query_model, maxfev=20000, print_res=False, object_type='generic'):
    """
    Estimates the rotor that takes one set of objects to another
    """
    grade_list = [grade_obj(q) for q in query_model]
    x0 = np.finfo(float).eps * np.random.rand(6)

    def minimisation_func(x):
        R = rotorconversion(x)
        query_model_remapped = [normalised((apply_rotor(l, R))(grade_list[i])) for i,l in enumerate(query_model)]
        return object_set_cost_sum(reference_model, query_model_remapped, object_type=object_type)

    res = minimize(minimisation_func, x0, method='SLSQP', options={'ftol': 10.0 ** (-16), \
                                                                      'maxiter': 1000, \
                                                                      'disp': False})
    if print_res:
        print(res)
    res = minimize(minimisation_func, res.x, method='L-BFGS-B', options={'ftol':10.0**(-16), \
                                                                       'maxiter':1000, \
                                                                       'disp': False, \
                                                                        'maxls':40})
    if print_res:
        print(res)
    rotor = rotorconversion(res.x)
    query_model_remapped = [normalised((apply_rotor(l, rotor))(grade_list[i])) for i,l in enumerate(query_model)]
    cost = object_set_cost_sum(reference_model, query_model_remapped, object_type=object_type)
    return rotor, cost


def cartans_lines(obj_list_a, obj_list_b):
    """ Performs the extended cartans algorithm as suggested by Alex Arsenovic """
    V_found,rs = cartan(A=obj_list_a,B=obj_list_b)
    theta = ((V_found*~V_found)*e1234)(0)
    V_found = e**(-theta/2*e123inf)*V_found
    return V_found


def sequential_object_rotor_estimation(reference_model, query_model, n_iterations=500,
                                       cost_tolerance=10*(10**-16), random_sequence=False,
                                       object_type='generic'):
    """
    Performs a sequential rotor update based on the rotors between individual objects
    Exits when the sum of the cost of rotor updates through the list is very small
    """
    grade_list = [grade_obj(q) for q in query_model]

    R_total = 1.0 + 0.0*e1
    for j in range(n_iterations):
        cost_sum = 0.0
        if random_sequence:
            indices = random.sample(range(len(query_model)), len(query_model))
        else:
            indices = range(len(query_model))
        for i in indices:
            C1 = normalised(apply_rotor(query_model[i], R_total)(grade_list[i]))
            C2 = reference_model[i]
            if abs(C1 + C2) < 0.0001:
                C1 = -C1
            if object_type == 'lines':
                rroot = normalised(square_roots_of_rotor(rotor_between_lines(C1, C2))[0])
            else:
                try:
                    rroot = normalised(square_roots_of_rotor(rotor_between_objects(C1, C2))[0])
                except:
                    print(C1)
                    print(C2)
                    print(rotor_between_objects(C1, C2))
                    rroot = normalised(square_roots_of_rotor(rotor_between_objects(C1, C2))[0])
            R_total = normalised(rroot*R_total)
            cost_sum += rotor_cost(rroot)
        if cost_sum < cost_tolerance:
            exit_flag = 0
            return R_total, exit_flag
    exit_flag = 1
    return R_total, exit_flag


def sequential_object_rotor_estimation_convergence_detection(reference_model, query_model, n_iterations=500,
                                                             cost_tolerance=10*(10**-16), random_sequence=False,
                                                             object_type='generic'):
    """
    Performs a sequential rotor update based on the rotors between individual objects
    Exits when a full rotation through all objects produces a very small update of rotor
    """
    grade_list = [grade_obj(ob) for ob in query_model]

    R_total = 1.0 + 0.0*e1
    for j in range(n_iterations):
        r_set = 1.0 + 0.0*e1
        if random_sequence:
            indices = random.sample(range(len(query_model)), len(query_model))
        else:
            indices = range(len(query_model))
        for i in indices:
            grade = grade_list[i]
            new_obj = normalised(apply_rotor(query_model[i],R_total)(grade))
            C1 = normalised(new_obj)
            C2 = reference_model[i]
            if abs(C1 + C2) < 0.0001:
                C1 = -C1
            if object_type == 'lines':
                rroot = normalised(square_roots_of_rotor((rotor_between_objects(C1,C2)))[0])
            else:
                rroot = normalised(square_roots_of_rotor((rotor_between_objects(C1,C2)))[0])
            r_set = normalised(rroot*r_set)
            R_total = normalised(rroot * R_total)
        if rotor_cost(r_set) < cost_tolerance:
            exit_flag = 0
            return R_total, exit_flag
    exit_flag = 1
    return R_total, exit_flag



