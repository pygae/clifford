"""
Numba support for MultiVector objects.

For now, this just supports .value wrapping / unwrapping
"""
import numba
import operator
import numpy as np
from numba.extending import NativeValue
import llvmlite.ir

try:
    # module locations as of numba 0.49.0
    import numba.np.numpy_support as _numpy_support
    from numba.core.imputils import impl_ret_borrowed, lower_constant
    from numba.core import cgutils, types
except ImportError:
    # module locations prior to numba 0.49.0
    import numba.numpy_support as _numpy_support
    from numba.targets.imputils import impl_ret_borrowed, lower_constant
    from numba import cgutils, types

from .._multivector import MultiVector

from ._layout import LayoutType
from ._overload_call import overload_call

__all__ = ['MultiVectorType']


class MultiVectorType(types.Type):
    def __init__(self, layout: LayoutType, dtype: types.DType):
        self.layout_type = layout
        self._scalar_type = dtype
        super().__init__(name='MultiVector({!r}, {!r})'.format(
            self.layout_type, self._scalar_type
        ))

    @property
    def key(self):
        return self.layout_type, self._scalar_type

    @property
    def value_type(self):
        return self._scalar_type[:]


# The docs say we should use register a function to determine the numba type
# with `@numba.extending.typeof_impl.register(MultiVector)`, but this is way
# too slow (https://github.com/numba/numba/issues/5839). Instead, we use the
# undocumented `_numba_type_` attribute, and use our own cache. In future
# this may need to be a weak cache, but for now the objects are tiny anyway.

@property
def _numba_type_(self):
    layout_type = self.layout._numba_type_

    cache = layout_type._cache
    dt = self.value.dtype

    # now use the dtype to key that cache.
    try:
        return cache[dt]
    except KeyError:
        # Computing and hashing `dtype_type` is slow, so we do not use it as a
        # hash key. The raw numpy dtype is much faster to use as a key.
        dtype_type = _numpy_support.from_dtype(dt)
        ret = cache[dt] = MultiVectorType(layout_type, dtype_type)
        return ret

MultiVector._numba_type_ = _numba_type_


@numba.extending.register_model(MultiVectorType)
class MultiVectorModel(numba.extending.models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ('layout', fe_type.layout_type),
            ('value', fe_type.value_type),
        ]
        super().__init__(dmm, fe_type, members)


@numba.extending.type_callable(MultiVector)
def type_MultiVector(context):
    def typer(layout, value):
        if isinstance(layout, LayoutType) and isinstance(value, types.Array):
            return MultiVectorType(layout, value.dtype)
    return typer


@numba.extending.lower_builtin(MultiVector, LayoutType, types.Any)
def impl_MultiVector(context, builder, sig, args):
    typ = sig.return_type
    layout, value = args
    mv = cgutils.create_struct_proxy(typ)(context, builder)
    mv.layout = layout
    mv.value = value
    return impl_ret_borrowed(context, builder, sig.return_type, mv._getvalue())


@lower_constant(MultiVectorType)
def lower_constant_MultiVector(context, builder, typ: MultiVectorType, pyval: MultiVector) -> llvmlite.ir.Value:
    mv = cgutils.create_struct_proxy(typ)(context, builder)
    mv.value = context.get_constant_generic(builder, typ.value_type, pyval.value)
    mv.layout = context.get_constant_generic(builder, typ.layout_type, pyval.layout)
    return mv._getvalue()


@numba.extending.unbox(MultiVectorType)
def unbox_MultiVector(typ: MultiVectorType, obj: MultiVector, c) -> NativeValue:
    value = c.pyapi.object_getattr_string(obj, "value")
    layout = c.pyapi.object_getattr_string(obj, "layout")
    mv = cgutils.create_struct_proxy(typ)(c.context, c.builder)
    mv.layout = c.unbox(typ.layout_type, layout).value
    mv.value = c.unbox(typ.value_type, value).value
    c.pyapi.decref(value)
    c.pyapi.decref(layout)
    is_error = cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return NativeValue(mv._getvalue(), is_error=is_error)


@numba.extending.box(MultiVectorType)
def box_MultiVector(typ: MultiVectorType, val: llvmlite.ir.Value, c) -> MultiVector:
    mv = cgutils.create_struct_proxy(typ)(c.context, c.builder, value=val)
    mv_obj = c.box(typ.value_type, mv.value)
    layout_obj = c.box(typ.layout_type, mv.layout)

    # All the examples use `c.pyapi.unserialize(c.pyapi.serialize_object(MultiVector))` here.
    # Doing so is much slower, as it incurs pickle. This is probably safe.
    class_obj_ptr = c.context.add_dynamic_addr(c.builder, id(MultiVector), info=MultiVector.__name__)
    class_obj = c.builder.bitcast(class_obj_ptr, c.pyapi.pyobj)
    res = c.pyapi.call_function_objargs(class_obj, (layout_obj, mv_obj))
    c.pyapi.decref(mv_obj)
    c.pyapi.decref(layout_obj)
    return res


numba.extending.make_attribute_wrapper(MultiVectorType, 'value', 'value')
numba.extending.make_attribute_wrapper(MultiVectorType, 'layout', 'layout')


@numba.extending.overload(operator.add)
def ga_add(a, b):
    if isinstance(a, MultiVectorType) and isinstance(b, MultiVectorType):
        if a.layout_type != b.layout_type:
            raise numba.TypingError("MultiVector objects belong to different layouts")
        def impl(a, b):
            return a.layout.MultiVector(a.value + b.value)
        return impl
    elif isinstance(a, types.abstract.Number) and isinstance(b, MultiVectorType):
        scalar_index = b.layout_type.obj._basis_blade_order.bitmap_to_index[0]
        ret_type = np.result_type(_numpy_support.as_dtype(a), _numpy_support.as_dtype(b.value_type.dtype))
        def impl(a, b):
            op = b.value.astype(ret_type)
            op[scalar_index] += a
            return b.layout.MultiVector(op)
        return impl
    elif isinstance(a, MultiVectorType) and isinstance(b, types.abstract.Number):
        scalar_index = a.layout_type.obj._basis_blade_order.bitmap_to_index[0]
        ret_type = np.result_type(_numpy_support.as_dtype(a.value_type.dtype), _numpy_support.as_dtype(b))
        def impl(a, b):
            op = a.value.astype(ret_type)
            op[scalar_index] += b
            return a.layout.MultiVector(op)
        return impl


@numba.extending.overload(operator.sub)
def ga_sub(a, b):
    if isinstance(a, MultiVectorType) and isinstance(b, MultiVectorType):
        if a.layout_type != b.layout_type:
            raise numba.TypingError("MultiVector objects belong to different layouts")
        def impl(a, b):
            return a.layout.MultiVector(a.value - b.value)
        return impl
    elif isinstance(a, types.abstract.Number) and isinstance(b, MultiVectorType):
        scalar_index = b.layout_type.obj._basis_blade_order.bitmap_to_index[0]
        ret_type = np.result_type(_numpy_support.as_dtype(a), _numpy_support.as_dtype(b.value_type.dtype))
        def impl(a, b):
            op = -b.value.astype(ret_type)
            op[scalar_index] += a
            return b.layout.MultiVector(op)
        return impl
    elif isinstance(a, MultiVectorType) and isinstance(b, types.abstract.Number):
        scalar_index = a.layout_type.obj._basis_blade_order.bitmap_to_index[0]
        ret_type = np.result_type(_numpy_support.as_dtype(a.value_type.dtype), _numpy_support.as_dtype(b))
        def impl(a, b):
            op = a.value.astype(ret_type)
            op[scalar_index] -= b
            return a.layout.MultiVector(op)
        return impl


@numba.extending.overload(operator.mul)
def ga_mul(a, b):
    if isinstance(a, MultiVectorType) and isinstance(b, MultiVectorType):
        if a.layout_type != b.layout_type:
            raise numba.TypingError("MultiVector objects belong to different layouts")
        gmt_func = a.layout_type.obj.gmt_func
        def impl(a, b):
            return a.layout.MultiVector(gmt_func(a.value, b.value))
        return impl
    elif isinstance(a, types.abstract.Number) and isinstance(b, MultiVectorType):
        def impl(a, b):
            return b.layout.MultiVector(a*b.value)
        return impl
    elif isinstance(a, MultiVectorType) and isinstance(b, types.abstract.Number):
        def impl(a, b):
            return a.layout.MultiVector(a.value*b)
        return impl


@numba.extending.overload(operator.xor)
def ga_xor(a, b):
    if isinstance(a, MultiVectorType) and isinstance(b, MultiVectorType):
        if a.layout_type != b.layout_type:
            raise numba.TypingError("MultiVector objects belong to different layouts")
        omt_func = a.layout_type.obj.omt_func
        def impl(a, b):
            return a.layout.MultiVector(omt_func(a.value, b.value))
        return impl
    elif isinstance(a, types.abstract.Number) and isinstance(b, MultiVectorType):
        def impl(a, b):
            return b.layout.MultiVector(b.value*a)
        return impl
    elif isinstance(a, MultiVectorType) and isinstance(b, types.abstract.Number):
        def impl(a, b):
            return a.layout.MultiVector(a.value*b)
        return impl

@numba.extending.overload(operator.or_)
def ga_or(a, b):
    if isinstance(a, MultiVectorType) and isinstance(b, MultiVectorType):
        if a.layout_type != b.layout_type:
            raise numba.TypingError("MultiVector objects belong to different layouts")
        imt_func = a.layout_type.obj.imt_func
        def impl(a, b):
            return a.layout.MultiVector(imt_func(a.value, b.value))
        return impl
    elif isinstance(a, types.abstract.Number) and isinstance(b, MultiVectorType):
        ret_type = np.result_type(_numpy_support.as_dtype(a), _numpy_support.as_dtype(b.value_type.dtype))
        def impl(a, b):
            return b.layout.MultiVector(np.zeros_like(b.value, dtype=ret_type))
        return impl
    elif isinstance(a, MultiVectorType) and isinstance(b, types.abstract.Number):
        ret_type = np.result_type(_numpy_support.as_dtype(a.value_type.dtype), _numpy_support.as_dtype(b))
        def impl(a, b):
            return a.layout.MultiVector(np.zeros_like(a.value, dtype=ret_type))
        return impl


@numba.extending.overload(operator.truediv)
def ga_truediv(a, b):
    if isinstance(a, MultiVectorType) and isinstance(b, types.abstract.Number):
        def impl(a, b):
            return a.layout.MultiVector(a.value / b)
        return impl
    # TODO: implement inversion for the other pairs


@numba.extending.overload(operator.invert)
def ga_invert(a):
    if isinstance(a, MultiVectorType):
        adjoint_func = a.layout_type.obj.adjoint_func
        def impl(a):
            return a.layout.MultiVector(adjoint_func(a.value))
        return impl


@numba.extending.overload(operator.pos)
def ga_pos(a):
    if isinstance(a, MultiVectorType):
        def impl(a):
            return a.layout.MultiVector(a.value.copy())
        return impl


@numba.extending.overload(operator.neg)
def ga_neg(a):
    if isinstance(a, MultiVectorType):
        def impl(a):
            return a.layout.MultiVector(-a.value)
        return impl


@overload_call(MultiVectorType)
def ga_call(self, arg):
    # grade projection
    grades = self.layout_type.obj._basis_blade_order.grades
    if isinstance(arg, types.IntegerLiteral):
        # Optimized case where the mask can be computed at compile-time.
        # using `nonzero` makes the resulting array smaller.
        inds, = (grades == arg.literal_value).nonzero()
        def impl(self, arg):
            mv = self.layout.MultiVector(np.zeros_like(self.value))
            mv.value[inds] = self.value[inds]
            return mv
        return impl
    elif isinstance(arg, types.Integer):
        # runtime grade selection - should be less common
        def impl(self, arg):
            # probably faster to not call nonzero here
            inds = grades == arg
            mv = self.layout.MultiVector(np.zeros_like(self.value))
            mv.value[inds] = self.value[inds]
            return mv
        return impl


@numba.extending.overload_method(MultiVectorType, 'mag2')
def MultiVector_mag2(self):
    def impl(self):
        return (~self * self).value[0]
    return impl


@numba.extending.overload(abs)
def MultiVector___abs__(self):
    if isinstance(self, MultiVectorType):
        return MultiVector.__abs__


@numba.extending.overload_method(MultiVectorType, 'normal')
def MultiVector_normal(self):
    return MultiVector.normal
