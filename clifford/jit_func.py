
import ast
import astpretty
import inspect
from ._numba_utils import njit
from ._ast_transformer import GATransformer


class jit_func(object):

    def __init__(self, layout, ast_debug=False):
        self.layout = layout
        self.ast_debug = ast_debug

    def __call__(self, func):
        # Get the function source
        fname = func.__name__
        source = inspect.getsource(func)
        source = '\n'.join(source.splitlines()[1:]) # remove the decorator first line.

        # Re-write the ast
        tree = ast.parse(source)
        if self.ast_debug:
            print('\n\n\n\n TRANFORMING FROM \n\n\n\n')
            astpretty.pprint(tree)

        tree = GATransformer().visit(tree)
        ast.fix_missing_locations(tree)

        if self.ast_debug:
            print('\n\n\n\n TRANFORMING TO \n\n\n\n')
            astpretty.pprint(tree)

        # Compile the function
        co = compile(tree, '<ast>', "exec")
        locals_dict = {}
        exec(co, globals(), locals_dict)
        new_func = locals_dict[fname]

        # Set things up into memory so that they JIT ok...
        as_ga = self.layout.as_ga_value_vector_func
        ga_add = self.layout.overload_add_func
        ga_sub = self.layout.overload_sub_func
        ga_mul = self.layout.overload_mul_func
        ga_xor = self.layout.overload_xor_func
        ga_or = self.layout.overload_or_func

        # globals()['as_ga'] = as_ga
        # globals()['ga_add'] = ga_add
        # globals()['ga_sub'] = ga_sub
        # globals()['ga_mul'] = ga_mul
        # globals()['ga_xor'] = ga_xor
        # globals()['ga_or'] = ga_or

        # JIT the function
        jitted_func = njit(new_func)

        # Wrap the JITed function
        def wrapper(*args, **kwargs):
            return self.layout.MultiVector(value=jitted_func(*[a.value for a in args], **kwargs))
        return wrapper
