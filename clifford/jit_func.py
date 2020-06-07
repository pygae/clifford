
import ast
try:
    import astpretty
    AST_PRETTY_AVAILABLE = True
except ImportError:
    AST_PRETTY_AVAILABLE = False
import inspect
import warnings
import textwrap
from ._numba_utils import njit
from ._ast_transformer import DecoratorRemover, GATransformer


class jit_func(object):
    """
    This is a JIT decorator that re-writes the AST and then numba JITs
    the resulting function.
    """
    def __init__(self, layout, ast_debug=False, mv_constants={}, scalar_constants={}):
        self.layout = layout
        self.mv_constants = mv_constants
        self.scalar_constants = scalar_constants
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
        tree = DecoratorRemover().visit(tree)
        # Re-write the ast
        tree = GATransformer().visit(tree)
        # Fix it all up
        ast.fix_missing_locations(tree)

        if self.ast_debug:
            print('\n\n\n\n TRANSFORMING TO \n\n\n\n')
            astpretty.pprint(tree)

        # Set things up into locals and globals so that they JIT ok...
        locals_dict = {'as_ga': self.layout.as_ga_value_vector_func,
                       'ga_add': self.layout.overload_add_func,
                       'ga_sub': self.layout.overload_sub_func,
                       'ga_mul': self.layout.overload_mul_func,
                       'ga_xor': self.layout.overload_xor_func,
                       'ga_or': self.layout.overload_or_func,
                       'ga_rev': self.layout.overload_reverse_func,
                       'ga_call': self.layout.overload_call_func}

        # Add the passed multivector and scalar constants
        for k, v in self.mv_constants.items():
            locals_dict[k] = v.value
        for k, v in self.scalar_constants.items():
            locals_dict[k] = v

        # TODO: Work out a better way to deal with changes to globals
        globs = {}
        for k, v in globals().items():
            globs[k] = v
        for k, v in locals_dict.items():
            globs[k] = v

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
