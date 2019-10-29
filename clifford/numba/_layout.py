import numba
import numba.extending
try:
    # module locations as of numba 0.49.0
    from numba.core import cgutils, types
except ImportError:
    # module locations prior to numba 0.49.0
    from numba import cgutils, types

from .._layout import Layout


# Taken from numba_passthru
opaque_pyobject = types.Opaque('Opaque(PyObject)')


class LayoutType(types.Type):
    def __init__(self):
        super().__init__("LayoutType")


@numba.extending.register_model(LayoutType)
class LayoutModel(numba.extending.models.StructModel):
    def __init__(self, dmm, fe_typ):
        members = [
            ('meminfo', types.MemInfoPointer(opaque_pyobject)),
        ]
        super().__init__(dmm, fe_typ, members)


@numba.extending.typeof_impl.register(Layout)
def _typeof_Layout(val: Layout, c) -> LayoutType:
    return LayoutType()


# Derived from numba_passthru

@numba.extending.unbox(LayoutType)
def unbox_Layout(typ, obj, context):
    layout = cgutils.create_struct_proxy(typ)(context.context, context.builder)
    layout.meminfo = context.pyapi.nrt_meminfo_new_from_pyobject(obj, obj)
    return numba.extending.NativeValue(layout._getvalue())


@numba.extending.box(LayoutType)
def box_Layout(typ, val, context):
    val = cgutils.create_struct_proxy(typ)(context.context, context.builder, value=val)
    obj = context.context.nrt.meminfo_data(context.builder, val.meminfo)
    context.pyapi.incref(obj)
    return obj
