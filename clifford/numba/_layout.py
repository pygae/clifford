import numba
import numba.extending
from numba.extending import NativeValue
import llvmlite.ir
try:
    # module locations as of numba 0.49.0
    from numba.core import cgutils, types
    from numba.core.imputils import lower_constant
except ImportError:
    # module locations prior to numba 0.49.0
    from numba import cgutils, types
    from numba.targets.imputils import lower_constant

from .._layout import Layout
from .._multivector import MultiVector


opaque_layout = types.Opaque('Opaque(Layout)')


class LayoutType(types.Type):
    def __init__(self):
        super().__init__("LayoutType")


@numba.extending.register_model(LayoutType)
class LayoutModel(numba.extending.models.StructModel):
    def __init__(self, dmm, fe_typ):
        members = [
            ('obj', opaque_layout),
        ]
        super().__init__(dmm, fe_typ, members)


@numba.extending.typeof_impl.register(Layout)
def _typeof_Layout(val: Layout, c) -> LayoutType:
    return LayoutType()


# Derived from the `Dispatcher` boxing

@lower_constant(LayoutType)
def lower_constant_Layout(context, builder, typ: LayoutType, pyval: Layout) -> llvmlite.ir.Value:
    layout = cgutils.create_struct_proxy(typ)(context, builder)
    layout.obj = context.add_dynamic_addr(builder, id(pyval), info=type(pyval).__name__)
    return layout._getvalue()


@numba.extending.unbox(LayoutType)
def unbox_Layout(typ: LayoutType, obj: Layout, c) -> NativeValue:
    layout = cgutils.create_struct_proxy(typ)(c.context, c.builder)
    layout.obj = obj
    return NativeValue(layout._getvalue())


@numba.extending.box(LayoutType)
def box_Layout(typ: LayoutType, val: llvmlite.ir.Value, c) -> Layout:
    val = cgutils.create_struct_proxy(typ)(c.context, c.builder, value=val)
    obj = val.obj
    c.pyapi.incref(obj)
    return obj

# methods

@numba.extending.overload_method(LayoutType, 'MultiVector')
def Layout_MultiVector(self, value):
    def impl(self, value):
        return MultiVector(self, value)
    return impl
