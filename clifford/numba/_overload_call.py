"""
Numba support for overloading the `__call__` operator.

This is a workaround until https://github.com/numba/numba/issues/5885 is
resolved.
"""
import numba
import numba.extending

try:
    # module locations as of numba 0.49.0
    from numba.core.typing.templates import (
        AbstractTemplate, _OverloadAttributeTemplate, make_overload_attribute_template)
    from numba.core import types
except ImportError:
    # module locations prior to numba 0.49.0
    from numba.typing.templates import (
        AbstractTemplate, _OverloadAttributeTemplate, make_overload_attribute_template)
    from numba import types

__all__ = ['overload_call']


class _OverloadCallTemplate(_OverloadAttributeTemplate):
    """
    Modified version of _OverloadMethodTemplate for overloading `__call__`.

    When typing, numba requires a `__call__` attribute to be provided as a
    `BoundFunction` instance.

    When lowering, the code in `numba.core.base.BaseContext.get_function`
    expects to find the implementation under the key `NumbaType` - but
    overload_method uses the key `(NumbaType, '__call__')`.

    The only change in this class is to fix up they keys.
    """
    is_method = True

    @classmethod
    def do_class_init(cls):
        """
        Register generic method implementation.
        """

        # this line is changed for __call__
        @numba.extending.lower_builtin(cls.key, cls.key, types.VarArg(types.Any))
        def method_impl(context, builder, sig, args):
            typ = sig.args[0]
            typing_context = context.typing_context
            fnty = cls._get_function_type(typing_context, typ)
            sig = cls._get_signature(typing_context, fnty, sig.args, {})
            call = context.get_function(fnty, sig)
            # Link dependent library
            context.add_linking_libs(getattr(call, 'libs', ()))
            return call(builder, args)

    def _resolve(self, typ, attr):
        if self._attr != attr:
            return None

        assert isinstance(typ, self.key)

        class MethodTemplate(AbstractTemplate):
            key = self.key  # this line is changed for __call__
            _inline = self._inline
            _overload_func = staticmethod(self._overload_func)
            _inline_overloads = self._inline_overloads

            def generic(_, args, kws):
                args = (typ,) + tuple(args)
                fnty = self._get_function_type(self.context, typ)
                sig = self._get_signature(self.context, fnty, args, kws)
                sig = sig.replace(pysig=numba.extending.utils.pysignature(self._overload_func))
                for template in fnty.templates:
                    self._inline_overloads.update(template._inline_overloads)
                if sig is not None:
                    return sig.as_method()

        return types.BoundFunction(MethodTemplate, typ)



def overload_call(typ, **kwargs):

    def decorate(overload_func):
        template = make_overload_attribute_template(
            typ, '__call__', overload_func,
            inline=kwargs.get('inline', 'never'),
            base=_OverloadCallTemplate
        )
        numba.extending.infer_getattr(template)
        numba.extending.overload(overload_func, **kwargs)(overload_func)
        return overload_func

    return decorate
