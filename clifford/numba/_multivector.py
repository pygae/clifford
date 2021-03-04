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
        self.value_type = dtype
        super().__init__(name='MultiVector({!r}, {!r})'.format(
            self.layout_type, self.value_type
        ))

    @property
    def key(self):
        return self.layout_type, self.value_type


# The docs say we should use register a function to determine the numba type
# with `@numba.extending.typeof_impl.register(MultiVector)`, but this is way
# too slow (https://github.com/numba/numba/issues/5839). Instead, we use the
# undocumented `_numba_type_` attribute, and use our own cache. In future
# this may need to be a weak cache, but for now the objects are tiny anyway.

@property
def _numba_type_(self):
    # If the array is not 1D we can't do anything with it
    if self.value.ndim != 1:
        return None

    layout_type = self.layout._numba_type_

    dt = self.value.dtype
    if self.value.flags.c_contiguous:
        relevant_cache = layout_type._c_cache
    else:
        relevant_cache = layout_type._a_cache

    # now use the dtype to key that cache.
    try:
        return relevant_cache[dt]
    except KeyError:
        # Computing and hashing `value_type` is slow, so we do not use it as a
        # hash key. The raw numpy dtype is much faster to use as a key.
        if self.value.flags.c_contiguous:
            value_type = _numpy_support.from_dtype(dt)[::1]
        else:
            value_type = _numpy_support.from_dtype(dt)[:]
        ret = relevant_cache[dt] = MultiVectorType(layout_type, value_type)
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
            return MultiVectorType(layout, value)
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


@numba.extending.overload(operator.pow)
def ga_pow(a, b):
    if isinstance(a, MultiVectorType) and isinstance(b, types.Integer):
        gmt_func = a.layout_type.obj.gmt_func
        def impl(a, b):
            if b < 0:
                raise NotImplementedError('Negative powers are currently not implemented')
            if b == 0:
                return 1 + 0*a
            op = a.value
            for i in range(1, b):
                op = gmt_func(op, a.value)
            return a.layout.MultiVector(op)
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
def ga_call(self, *args):
    # a numba quirk means that varargs can end up passed in two different ways
    if len(args) == 1 and isinstance(args[0], (types.StarArgTuple, types.StarArgUniTuple)):
        args = args[0].types

    # grade selection
    if len(args) > 0:
        # grade projection
        grades = self.layout_type.obj._basis_blade_order.grades
        if all(isinstance(arg, types.IntegerLiteral) for arg in args):
            # Optimized case where the mask can be computed at compile-time.
            inds = (grades == args[0].literal_value)
            for arg in args[1:]:
                inds |= (grades == arg.literal_value)
            # using `nonzero` makes the resulting array smaller.
            inds = inds.nonzero()
            def impl(self, *args):
                mv = self.layout.MultiVector(np.zeros_like(self.value))
                mv.value[inds] = self.value[inds]
                return mv
            return impl
        elif all(isinstance(arg, types.Integer) for arg in args):
            # runtime grade selection - should be less common. This includes
            # the case where only some grades are known at compile-time.
            def impl(self, *args):
                # probably faster to not call nonzero here
                inds = (grades == args[0])
                # can't use `for arg in args` here due to numba/numba#5372
                for i in range(1, len(args)):
                    inds |= (grades == args[i])
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


@numba.extending.overload_method(MultiVectorType, 'gradeInvol')
def MultiVector_gradeInvol(self):
    g_func = self.layout_type.obj._grade_invol
    def impl(self):
        return g_func(self)
    return impl


@numba.extending.overload_method(MultiVectorType, 'conjugate')
def MultiVector_conjugate(self):
    return MultiVector.conjugate
