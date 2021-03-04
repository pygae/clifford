import numba
import numba.extending
from numba.extending import NativeValue
import llvmlite.ir
try:
    # module locations as of numba 0.49.0
    from numba.core import types
    from numba.core.imputils import lower_constant
except ImportError:
    # module locations prior to numba 0.49.0
    from numba import types
    from numba.targets.imputils import lower_constant

from .._layout import Layout, _cached_property
from .._layout_helpers import layout_short_name
from .._multivector import MultiVector


# In future we want to store some of the layout in the type (the `order` etc),
# but store the `names` in the layout instances, so that we can reuse jitted
# functions across different basis vector names.

class LayoutType(types.Dummy):
    def __init__(self, layout):
        self.obj = layout
        # Caches of multivector types for this layout, in numba C and A order.
        # Having two caches is faster than a cache keyed by a tuple of `(order, dt)`,
        # and every millisecond counts in `MultiVector._numba_type_`.
        self._c_cache = {}
        self._a_cache = {}
        layout_name = layout_short_name(layout)
        if layout_name is not None:
            name = "LayoutType({})".format(layout_name)
        else:
            name = "LayoutType({!r})".format(layout)
        super().__init__(name)

    def __getstate__(self):
        # the cache causes issues with numba's pickle modifications, as it
        # contains a self-reference.
        d = self.__dict__.copy()
        d['_c_cache'] = {}
        d['_a_cache'] = {}
        return d


@numba.extending.register_model(LayoutType)
class LayoutModel(numba.extending.models.OpaqueModel):
    pass

# The docs say we should use register a function to determine the numba type
# with `@numba.extending.typeof_impl.register(LayoutType)`, but this is way
# too slow (https://github.com/numba/numba/issues/5839). Instead, we use the
# undocumented `_numba_type_` attribute, and use our own cache.

@_cached_property
def _numba_type_(self):
    return LayoutType(self)

Layout._numba_type_ = _numba_type_


@lower_constant(LayoutType)
def lower_constant_Layout(context, builder, typ: LayoutType, pyval: Layout) -> llvmlite.ir.Value:
    return context.get_dummy_value()


@numba.extending.unbox(LayoutType)
def unbox_Layout(typ: LayoutType, obj: Layout, c) -> NativeValue:
    return NativeValue(c.context.get_dummy_value())

# Derived from the `Dispatcher` boxing

@numba.extending.box(LayoutType)
def box_Layout(typ: LayoutType, val: llvmlite.ir.Value, c) -> Layout:
    obj = c.context.add_dynamic_addr(c.builder, id(typ.obj), info=typ.name)
    c.pyapi.incref(obj)
    return obj

# methods

@numba.extending.overload_method(LayoutType, 'MultiVector')
def Layout_MultiVector(self, value):
    def impl(self, value):
        return MultiVector(self, value)
    return impl
