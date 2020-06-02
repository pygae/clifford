"""
Helpers to allow tests to pass when ``NUMBA_DISABLE_JIT`` is set.

There are two issues to deal with here:

* Disabling jit makes things not pickleable that usually would be.
* `generated_jit` does not have sensible behavior in this mode.
  (https://github.com/numba/numba/issues/4759)

In the long run, we may also want to look into allowing numba to not be
installed at all, for uses like pyiodide.
"""
import functools

import numba

try:
    from numba.core.config import DISABLE_JIT
    import numba.core.serialize as _serialize
except ImportError:
    from numba.config import DISABLE_JIT
    import numba.serialize as _serialize


class pickleable_function:
    """
    Numba jitted functions are pickleable, so we should be too.

    Here we just reach into the internals of numba to pull out their
    serialization helpers
    """
    def __new__(cls, func):
        if isinstance(func, pickleable_function):
            return func
        self = super().__new__(cls)
        self.__func = func
        functools.update_wrapper(self, self.__func)
        return self

    @classmethod
    def _rebuild(cls, *args):
        return cls(_serialize._rebuild_function(*args))

    def __reduce__(self):
        globs = _serialize._get_function_globals_for_reduction(self.__func)
        return (
            self._rebuild,
            _serialize._reduce_function(self.__func, globs)
        )

    def __call__(self, *args, **kwargs):
        return self.__func(*args, **kwargs)

    def __repr__(self):
        return "_pickleable_function({!r})".format(self.__func)


class _fake_generated_jit:
    def __init__(self, f):
        self.__cache = {}
        self.__func = pickleable_function(f)
        functools.update_wrapper(self, self.__func)

    def __getnewargs_ex__(self):
        return (self.__func,), {}

    def __getstate__(self):
        return {}

    def __call__(self, *args):
        arg_type = tuple(numba.typeof(arg) for arg in args)
        try:
            func = self.__cache[arg_type]
        except KeyError:
            func = self.__cache[arg_type] = self.__func(*arg_type)
        return func(*args)


if not DISABLE_JIT:
    njit = numba.njit
    generated_jit = numba.generated_jit
else:
    def njit(f=None, **kwargs):
        if f is None:
            return pickleable_function
        else:
            return pickleable_function(f)

    def generated_jit(f=None, **kwargs):
        if f is None:
            return _fake_generated_jit
        else:
            return _fake_generated_jit(f)
