
import ast
try:
    import astpretty
    AST_PRETTY_AVAILABLE = True
except ImportError:
    AST_PRETTY_AVAILABLE = False
import inspect
import warnings
import textwrap
import weakref
import functools
import numpy as np
from numba.extending import overload
from numba import types
from ._numba_utils import njit
from ._ast_transformer import DecoratorRemover, GATransformer


def get_as_ga_vector_func(layout):
    """
    Returns a function that converts a scalar into a GA value vector
    for the given algebra
    """
    scalar_index = layout._basis_blade_order.bitmap_to_index[0]
    ndims = layout.gaDims
    @njit
    def as_ga_value_vector(x):
        op = np.zeros(ndims)
        op[scalar_index] = x
        return op
    return as_ga_value_vector


def get_overload_add_func(layout):
    """
    Returns an overloaded JITed function that works on
    MultiVector value arrays
    """
    scalar_index = layout._basis_blade_order.bitmap_to_index[0]

    def ga_add(a, b):
        # dummy function to overload
        pass

    @overload(ga_add, inline='always')
    def ol_ga_add(a, b):
        if isinstance(a, types.abstract.Number) and isinstance(b, types.Array):
            def impl(a, b):
                op = b.astype(np.float32)
                op[scalar_index] += a
                return op
            return impl
        elif isinstance(a, types.Array) and isinstance(b, types.abstract.Number):
            def impl(a, b):
                op = a.astype(np.float32)
                op[scalar_index] += b
                return op
            return impl
        else:
            def impl(a, b):
                return a + b
            return impl

    return ga_add


def get_overload_sub_func(layout):
    """
    Returns an overloaded JITed function that works on
    MultiVector value arrays
    """
    scalar_index = layout._basis_blade_order.bitmap_to_index[0]

    def ga_sub(a, b):
        # dummy function to overload
        pass

    @overload(ga_sub, inline='always')
    def ol_ga_sub(a, b):
        if isinstance(a, types.abstract.Number) and isinstance(b, types.Array):
            def impl(a, b):
                op = -b.astype(np.float32)
                op[scalar_index] += a
                return op
            return impl
        elif isinstance(a, types.Array) and isinstance(b, types.abstract.Number):
            def impl(a, b):
                op = a.astype(np.float32)
                op[scalar_index] -= b
                return op
            return impl
        else:
            def impl(a, b):
                return a - b
            return impl

    return ga_sub


def get_overload_mul_func(layout):
    """
    Returns an overloaded JITed function that works on
    MultiVector value arrays
    """
    def ga_mul(a, b):
        # dummy function to overload
        pass

    gmt_func = layout.gmt_func
    @overload(ga_mul, inline='always')
    def ol_ga_mul(a, b):
        if isinstance(a, types.Array) and isinstance(b, types.Array):
            def impl(a, b):
                return gmt_func(a, b)
            return impl
        else:
            def impl(a, b):
                return a*b
            return impl

    return ga_mul


def get_overload_xor_func(layout):
    """
    Returns an overloaded JITed function that works on
    MultiVector value arrays
    """
    def ga_xor(a, b):
        # dummy function to overload
        pass

    as_ga = get_as_ga_vector_func(layout)
    omt_func = layout.omt_func
    @overload(ga_xor, inline='always')
    def ol_ga_xor(a, b):
        if isinstance(a, types.Array) and isinstance(b, types.Array):
            def impl(a, b):
                return omt_func(a, b)
            return impl
        elif isinstance(a, types.Array) and isinstance(b, types.abstract.Number):
            def impl(a, b):
                return omt_func(a, as_ga(b))
            return impl
        elif isinstance(a, types.abstract.Number) and isinstance(b, types.Array):
            def impl(a, b):
                return omt_func(as_ga(a), b)
            return impl
        else:
            def impl(a, b):
                return a^b
            return impl

    return ga_xor


def get_overload_or_func(layout):
    """
    Returns an overloaded JITed function that works on
    MultiVector value arrays
    """
    def ga_or(a, b):
        # dummy function to overload
        pass

    as_ga = get_as_ga_vector_func(layout)
    imt_func = layout.imt_func
    @overload(ga_or, inline='always')
    def ol_ga_or(a, b):
        if isinstance(a, types.Array) and isinstance(b, types.Array):
            def impl(a, b):
                return imt_func(a, b)
            return impl
        elif isinstance(a, types.Array) and isinstance(b, types.abstract.Number):
            def impl(a, b):
                return imt_func(a, as_ga(b))
            return impl
        elif isinstance(a, types.abstract.Number) and isinstance(b, types.Array):
            def impl(a, b):
                return imt_func(as_ga(a), b)
            return impl
        else:
            def impl(a, b):
                return a|b
            return impl

    return ga_or


def get_overload_reverse_func(layout):
    """
    Returns an overloaded JITed function that works on
    MultiVector value arrays
    """
    def ga_rev(x):
        # dummy function to overload
        pass

    adjoint_func = layout.adjoint_func
    @overload(ga_rev, inline='always')
    def ol_ga_rev(x):
        if isinstance(x, types.Array):
            def impl(x):
                return adjoint_func(x)
            return impl
        else:
            def impl(x):
                return ~x
            return impl

    return ga_rev


def get_project_to_grade_func(layout):
    """
    Returns a function that projects a multivector to a given grade
    """
    gradeList = np.array(layout.gradeList, dtype=int)
    ndims = layout.gaDims
    @njit
    def project_to_grade(A, g):
        op = np.zeros(ndims)
        for i in range(ndims):
            if gradeList[i] == g:
                op[i] = A[i]
        return op
    return project_to_grade


def get_overload_call_func(layout):
    """
    Returns an overloaded JITed function that works on
    MultiVector value arrays
    """
    def ga_call(a, b):
        # dummy function to overload
        pass

    project_to_grade = get_project_to_grade_func(layout)
    @overload(ga_call, inline='always')
    def ol_ga_call(a, b):
        if isinstance(a, types.Array) and isinstance(b, types.Integer):
            def impl(a, b):
                return project_to_grade(a, b)
            return impl
        else:
            def impl(a, b):
                return a(b)
            return impl

    return ga_call


def weak_cache(f):
    _cache = weakref.WeakKeyDictionary()
    @functools.wraps(f)
    def wrapped(*args, **kwargs):
        a, *args = args
        try:
            return _cache[a]
        except KeyError:
            ret =_cache[a] = f(a, *args, **kwargs)
            return ret
    wrapped._cache = _cache
    return wrapped


@weak_cache
def _get_jit_impls(layout):
    return {
        'as_ga': get_as_ga_vector_func(layout),
        'ga_add': get_overload_add_func(layout),
        'ga_sub': get_overload_sub_func(layout),
        'ga_mul': get_overload_mul_func(layout),
        'ga_xor': get_overload_xor_func(layout),
        'ga_or': get_overload_or_func(layout),
        'ga_rev': get_overload_reverse_func(layout),
        'ga_call': get_overload_call_func(layout),
    }


class jit_func(object):
    """
    This is a JIT decorator that re-writes the AST and then numba JITs
    the resulting function.
    """
    def __init__(self, layout, ast_debug=False, mv_constants={}, scalar_constants={}):
        self.layout = layout
        self.mv_constants = mv_constants
        self.scalar_constants = scalar_constants
        self.jit_impls = _get_jit_impls(layout)
        if AST_PRETTY_AVAILABLE:
            self.ast_debug = ast_debug
        else:
            if ast_debug:
                warnings.warn(
                    'The ast_debug flag is set to True, but the astpretty module is not importable.\n'
                    'To see ast_debug output please pip install astpretty')
            self.ast_debug = False

    def __call__(self, func):
        # Get the function source
        fname = func.__name__
        source = inspect.getsource(func)
        # Remove the indentation
        source = textwrap.dedent(source)

        # Parse the source
        tree = ast.parse(source)
        if self.ast_debug:
            print('\n\n\n\n TRANSFORMING FROM \n\n\n\n')
            astpretty.pprint(tree)

        # Remove the decorators from the function
        # TODO: Work out how to remove only the jit_func decorator
        tree = DecoratorRemover().visit(tree)
        # Re-write the ast
        tree = GATransformer().visit(tree)
        # Fix it all up
        ast.fix_missing_locations(tree)

        if self.ast_debug:
            print('\n\n\n\n TRANSFORMING TO \n\n\n\n')
            astpretty.pprint(tree)

        # Set things up into locals and globals so that they JIT ok...
        locals_dict = {}
        locals_dict.update(self.jit_impls)

        # Add the passed multivector and scalar constants
        locals_dict.update({k: v.value for k, v in self.mv_constants.items()})
        locals_dict.update(self.scalar_constants)

        # TODO: Work out a better way to deal with changes to globals
        globs = {}
        globs.update(globals())
        globs.update(locals_dict)

        # Compile the function
        co = compile(tree, '<ast>', "exec")
        exec(co, globs, locals_dict)
        new_func = locals_dict[fname]

        # JIT the function
        jitted_func = njit(new_func)

        # Wrap the JITed function
        def wrapper(*args, **kwargs):
            return self.layout.MultiVector(value=jitted_func(*[a.value for a in args], **kwargs))
        wrapper.value = jitted_func
        return wrapper
