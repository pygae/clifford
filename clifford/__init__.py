"""
.. currentmodule:: clifford

========================================
clifford (:mod:`clifford`)
========================================

The top-level module.
Provides two core classes, :class:`Layout` and :class:`MultiVector`, along with several helper functions to implement the algebras.

Constructing algebras
=====================

Note that typically the :doc:`predefined-algebras` are sufficient, and there is no need to build an algebra from scratch.

.. autosummary::
    :toctree:

    Cl
    conformalize

Whether you construct your algebras from scratch, or use the predefined ones, you'll end up working with the following types:

.. autosummary::
    :toctree:

    MultiVector
    Layout
    ConformalLayout

Advanced algebra configuration
------------------------------
It is unlikely you will need these features, but they remain as a better
spelling for features which have always been in ``clifford``.

.. autosummary::
    :toctree: generated/

    BasisBladeOrder
    BasisVectorIds

Global configuration functions
==============================
These functions are used to change the global behavior of ``clifford``.

.. autofunction:: eps

.. autofunction:: pretty

.. autofunction:: ugly

.. autofunction:: print_precision


Miscellaneous classes
=======================

.. autosummary::
    :toctree:

    MVArray
    Frame
    BladeMap

Miscellaneous functions
=======================

.. autosummary::
    :toctree:

    grade_obj
    randomMV

"""

# Standard library imports.
import os
import itertools
import warnings
from typing import List, Tuple, Set

# Major library imports.
import numpy as np
import numba as _numba  # to avoid clashing with clifford.numba
import sparse
try:
    from numba.np import numpy_support as _numpy_support
except ImportError:
    import numba.numpy_support as _numpy_support


from clifford.io import write_ga_file, read_ga_file  # noqa: F401

from ._version import __version__  # noqa: F401
from . import _numba_utils

from ._settings import pretty, ugly, eps, print_precision  # noqa: F401
import clifford.taylor_expansions as taylor_expansions

# For backwards-compatibility. New code should import directly from `clifford.operator`
from .operator import gp, op, ip  # noqa: F401

try:
    NUMBA_DISABLE_PARALLEL = os.environ['NUMBA_DISABLE_PARALLEL']
except KeyError:
    NUMBA_PARALLEL = True
else:
    NUMBA_PARALLEL = not bool(NUMBA_DISABLE_PARALLEL)


def general_exp(x, **kwargs):
    warnings.warn("cf.general_exp is deprecated. Use `mv.exp()` or `np.exp(mv)` on multivectors, or `cf.taylor_expansions.exp(x)` on arbitrary objects", DeprecationWarning, stacklevel=2)
    return taylor_expansions.exp(x, **kwargs)


def linear_operator_as_matrix(func, input_blades, output_blades):
    """
    Return a matrix that performs the operation of the provided linear
    operator function func mapping the input blades to the output blades
    """
    ndimin = len(input_blades)
    ndimout = len(output_blades)
    mat = np.zeros((ndimout, ndimin))
    for i, b in enumerate(input_blades):
        b_result = func(b)
        mat[:, i] = np.array([b_result[j] for j in output_blades])
    return mat


def get_mult_function(mt: sparse.COO, gradeList,
                      grades_a=None, grades_b=None, filter_mask=None):
    '''
    Returns a function that implements the mult_table on two input multivectors
    '''
    if (filter_mask is None) and (grades_a is not None) and (grades_b is not None):
        # If not specified explicitly, we can specify sparseness by grade
        filter_mask = np.zeros(mt.nnz, dtype=bool)
        k_list, _, m_list = mt.coords
        for i in range(len(filter_mask)):
            if gradeList[k_list[i]] in grades_a:
                if gradeList[m_list[i]] in grades_b:
                    filter_mask[i] = 1
        filter_mask = sparse.COO(coords=mt.coords, data=filter_mask, shape=mt.shape)

    if filter_mask is not None:
        # We can pass the sparse filter mask directly
        mt = sparse.where(filter_mask, mt, mt.dtype.type(0))

        return _get_mult_function(mt)

    else:
        return _get_mult_function_runtime_sparse(mt)


def _get_mult_function_result_type(a: _numba.types.Type, b: _numba.types.Type, mt: np.dtype):
    a_dt = _numpy_support.as_dtype(getattr(a, 'dtype', a))
    b_dt = _numpy_support.as_dtype(getattr(b, 'dtype', b))
    return np.result_type(a_dt, mt, b_dt)


def _get_mult_function(mt: sparse.COO):
    """
    Get a function similar to `` lambda a, b: np.einsum('i,ijk,k->j', a, mt, b)``

    Returns
    -------
    func : function (array_like (n_dims,), array_like (n_dims,)) -> array_like (n_dims,)
        A function that computes the appropriate multiplication
    """
    # unpack for numba
    dims = mt.shape[1]
    k_list, l_list, m_list = mt.coords
    mult_table_vals = mt.data

    @_numba_utils.generated_jit(nopython=True)
    def mv_mult(value, other_value):
        # this casting will be done at jit-time
        ret_dtype = _get_mult_function_result_type(value, other_value, mult_table_vals.dtype)
        mult_table_vals_t = mult_table_vals.astype(ret_dtype)

        def mult_inner(value, other_value):
            output = np.zeros(dims, dtype=ret_dtype)
            for k, l, m, val in zip(k_list, l_list, m_list, mult_table_vals_t):
                output[l] += value[k] * val * other_value[m]
            return output

        return mult_inner

    return mv_mult


def _get_mult_function_runtime_sparse(mt: sparse.COO):
    """
    A variant of `_get_mult_function` that attempts to exploit runtime zeros

    The returned function avoids performing multiplications if vectors contain
    zeros.

    TODO: determine if this actually helps.
    """
    # unpack for numba
    dims = mt.shape[1]
    k_list, l_list, m_list = mt.coords
    mult_table_vals = mt.data

    @_numba_utils.generated_jit(nopython=True)
    def mv_mult(value, other_value):
        # this casting will be done at jit-time
        ret_dtype = _get_mult_function_result_type(value, other_value, mult_table_vals.dtype)
        mult_table_vals_t = mult_table_vals.astype(ret_dtype)

        def mult_inner(value, other_value):
            output = np.zeros(dims, dtype=ret_dtype)
            for ind, k in enumerate(k_list):
                v_val = value[k]
                if v_val != 0.0:
                    m = m_list[ind]
                    ov_val = other_value[m]
                    if ov_val != 0.0:
                        l = l_list[ind]
                        output[l] += v_val * mult_table_vals_t[ind] * ov_val
            return output
        return mult_inner

    return mv_mult


@_numba_utils.njit
def grade_obj_func(objin_val, gradeList, threshold):
    """ returns the modal grade of a multivector """
    modal_value_count = np.zeros(objin_val.shape)
    n = 0
    for g in gradeList:
        if np.abs(objin_val[n]) > threshold:
            modal_value_count[g] += 1
        n += 1
    return np.argmax(modal_value_count)


def grade_obj(objin, threshold=0.0000001):
    '''
    Returns the modal grade of a multivector
    '''
    return grade_obj_func(objin.value, objin.layout._basis_blade_order.grades, threshold)


def grades_present(objin: 'MultiVector', threshold=0.0000001) -> Set[int]:
    # for backwards compatibility
    warnings.warn(
        "`clifford.grades_present(x)` is deprecated, use `x.grades()` instead. "
        "Note that the method uses `clifford.eps()` as the default tolerance.",
        DeprecationWarning, stacklevel=2)
    return objin.grades(eps=threshold)


# todo: work out how to let numba use the COO objects directly
@_numba_utils.njit
def _numba_val_get_left_gmt_matrix(x, k_list, l_list, m_list, mult_table_vals, ndims):
    # TODO: consider `dtype=result_type(x.dtype, mult_table_vals.dtype)`
    intermed = np.zeros((ndims, ndims), dtype=x.dtype)
    test_ind = 0
    for k in k_list:
        j = l_list[test_ind]
        i = m_list[test_ind]
        intermed[j, i] += mult_table_vals[test_ind] * x[k]
        test_ind = test_ind + 1
    return intermed


def val_get_left_gmt_matrix(mt: sparse.COO, x):
    """
    This produces the matrix X that performs left multiplication with x
    eg. X@b == (x*b).value
    """
    dims = mt.shape[1]
    k_list, l_list, m_list = mt.coords
    return _numba_val_get_left_gmt_matrix(
        x, k_list, l_list, m_list, mt.data, dims
    )


def val_get_right_gmt_matrix(mt: sparse.COO, x):
    """
    This produces the matrix X that performs right multiplication with x
    eg. X@b == (b*x).value
    """
    return val_get_left_gmt_matrix(mt.T, x)


# TODO: Move this to the top once we remove circular imports
from ._layout import Layout  # noqa: E402
from ._multivector import MultiVector  # noqa: E402
from ._conformal_layout import ConformalLayout  # noqa: E402
from ._layout_helpers import BasisVectorIds, BasisBladeOrder  # noqa: F401
from ._mvarray import MVArray, array  # noqa: F401
from ._frame import Frame  # noqa: F401

# this registers the extension type
from . import numba  # noqa: F401
from ._blademap import BladeMap  # noqa: F401


# copied from the itertools docs
def _powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return itertools.chain.from_iterable(
        itertools.combinations(s, r) for r in range(len(s) + 1)
    )


def elements(dims: int, firstIdx=0) -> List[Tuple[int, ...]]:
    """Return a list of tuples representing all 2**dims of blades
    in a dims-dimensional GA.

    Elements are sorted lexicographically.
    """
    return list(_powerset(range(firstIdx, firstIdx + dims)))


def Cl(p=0, q=0, r=0, sig=None, names=None, firstIdx=1, mvClass=MultiVector):
    r"""Returns a :class:`Layout` and basis blade :class:`MultiVector`\ s for the geometric algebra :math:`Cl_{p,q,r}`.

    The notation :math:`Cl_{p,q,r}` means that the algebra is :math:`p+q+r`-dimensional, with the first :math:`p` vectors with positive signature, the next :math:`q` vectors negative, and the final :math:`r` vectors with null signature.

    Returns
    =======
    layout : Layout
        The resulting layout
    blades : Dict[str, MultiVector]
        The blades of the returned layout, equivalent to ``layout.blades``.
    """
    if sig is None:
        layout = Layout._from_Cl(p, q, r, firstIdx=firstIdx, names=names)
    else:
        layout = Layout._from_sig(sig, firstIdx=firstIdx, names=names)
    return layout, layout.bases(mvClass=mvClass)


def bases(layout, *args, **kwargs):
    return layout.bases(*args, **kwargs)


def basis_vectors(layout):
    return layout.basis_vectors


def randomMV(
        layout, min=-2.0, max=2.0, grades=None, mvClass=MultiVector,
        uniform=None, n=1, normed=False, rng=None):
    """n Random MultiVectors with given layout.

    Coefficients are between min and max, and if grades is a list of integers,
    only those grades will be non-zero.


    Examples
    --------
    >>> randomMV(layout, min=-2.0, max=2.0, grades=None, uniform=None, n=2)  # doctest: +SKIP

    """

    if n > 1:
        # return many multivectors
        return [randomMV(layout=layout, min=min, max=max, grades=grades,
                         mvClass=mvClass, uniform=uniform, n=1,
                         normed=normed) for k in range(n)]

    if uniform is None:
        rng = np.random.default_rng(rng)
        uniform = rng.uniform

    if grades is None:
        mv = mvClass(layout, uniform(min, max, (layout.gaDims,)))
    else:
        if isinstance(grades, int):
            grades = [grades]
        newValue = np.zeros((layout.gaDims,))
        for i in range(layout.gaDims):
            if layout._basis_blade_order.grades[i] in grades:
                newValue[i] = uniform(min, max)
        mv = mvClass(layout, newValue)

    if normed:
        mv = mv.normal()

    return mv


def conformalize(layout, added_sig=[1, -1], *, mvClass=MultiVector, **kwargs):
    '''
    Conformalize a Geometric Algebra

    Given the `Layout` for a GA of signature (p, q), this
    will produce a GA of signature (p+1, q+1), as well as
    return a new list of blades and some `stuff`. `stuff`
    is a dict containing the null basis blades, and some
    up/down functions for projecting in/out of the CGA.



    Parameters
    -------------
    layout: `clifford.Layout`
        layout of the GA to conformalize (the base)
    added_sig: list-like
        list of +1, -1  denoted the added signatures
    **kwargs :
        passed to Cl() used to generate conformal layout

    Returns
    ---------
    layout_c : :class:`ConformalLayout`
        layout of the base GA
    blades_c : dict
        blades for the CGA
    stuff: dict
        dict mapping the following members of :class:`ConformalLayout` by their
        names, for easy unpacking into the global namespace:

        .. autosummary::

            ~ConformalLayout.ep
            ~ConformalLayout.en
            ~ConformalLayout.eo
            ~ConformalLayout.einf
            ~ConformalLayout.E0
            ~ConformalLayout.I_base
            ~ConformalLayout.up
            ~ConformalLayout.down
            ~ConformalLayout.homo


    Examples
    ---------
    >>> from clifford import Cl, conformalize
    >>> G2, blades = Cl(2)
    >>> G2c, bladesc, stuff = conformalize(G2)
    >>> locals().update(bladesc)
    >>> locals().update(stuff)
    '''
    layout_c = ConformalLayout._from_base_layout(layout, added_sig, **kwargs)
    stuff = {
        attr: getattr(layout_c, attr)
        for attr in [
            "ep", "en", "eo", "einf", "E0",
            "up", "down", "homo", "I_base",
        ]
    }
    return layout_c, layout_c.bases(mvClass=mvClass), stuff


# TODO: fix caching to work
# generate pre-defined algebras and cache them

# sigs = [(1, 1, 0), (2, 0, 0), (3, 1, 0), (3, 0, 0), (3, 2, 0), (4, 0, 0)]
# current_module = sys.modules[__name__]
# caching.build_or_read_cache_and_attach_submods(current_module, sigs=sigs)
